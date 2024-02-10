import os
import time
import json
import random
import logging
import argparse
import socketio
import numpy as np
from model.classification_model_wrapper import Models
from utils.model_dump import *

logging.getLogger('socketIO-client').setLevel(logging.WARNING)
random.seed(2018)
datestr = time.strftime('%m%d')
log_dir = os.path.join('experiments', 'logs', datestr)
if not os.path.exists(log_dir):
    raise FileNotFoundError("{} not found".format(log_dir))


def load_json(filename):
    """
    load a json object from a json file
    """
    with open(filename) as f:
        return json.load(f)


class LocalModel(object):
    """
    instantiate a local model class
    """

    def __init__(self, task_config):
        self.model_name = task_config['model_name']
        self.epoch = task_config['local_epoch']
        self.model = getattr(Models, self.model_name)(task_config)

    def get_weights(self):
        """
        gets the weights of the local model
        """
        return self.model.get_weights()

    def set_weights(self, new_weights):
        """
        sets the weights of the local model
        """
        self.model.set_weights(new_weights)

    def train_one_round(self):
        """
        federated round training
        """
        losses = []
        for i in range(1, self.epoch + 1):
            loss = self.model.train_one_epoch()
            losses.append(loss)
        return self.model.get_weights(), sum(losses) / len(losses)

    def evaluate(self):
        """
        evaluate the local model
        """
        accu, test_loss = self.model.evaluate()
        return accu, test_loss


class FederatedClient(object):
    """
    client side communication with the server
    """
    MAX_DATASET_SIZE_KEPT = 6000

    def __init__(self, server_host, server_port, task_config_filename,
                 gpu, ignore_load):
        USE_CPU = os.getenv('USE_CPU')
        if USE_CPU != "TRUE":
            os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu
        self.task_config = load_json(task_config_filename)
        self.ignore_load = ignore_load
        self.local_model = None
        self.dataset = None
        self.log_filename = self.task_config['log_filename']
        self.logger = logging.getLogger("client")
        self.fh = logging.FileHandler(os.path.join(log_dir, os.path.basename(self.log_filename)))
        self.fh.setLevel(logging.INFO)
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.ERROR)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(self.formatter)
        self.ch.setFormatter(self.formatter)
        self.logger.addHandler(self.fh)
        self.logger.addHandler(self.ch)
        self.logger.info(self.task_config)
        self.sio = socketio.Client()
        self.sio.connect(str(server_host) + ':' + str(server_port))
        self.register_handles()
        print("sent wakeup")
        self.sio.emit('client_wake_up')
        self.sio.wait()

    def on_init(self):
        print('on init')
        self.logger.info(self.task_config)
        self.local_model = LocalModel(self.task_config)
        print("local model initialized done.")
        self.sio.emit('client_ready')

    def load_stat(self):
        loadavg = {}
        with open("/proc/loadavg") as fin:
            con = fin.read().split()
            loadavg['lavg_1'] = con[0]
            loadavg['lavg_5'] = con[1]
            loadavg['lavg_15'] = con[2]
            loadavg['nr'] = con[3]
            loadavg['last_pid'] = con[4]
        return loadavg['lavg_15']

    def register_handles(self):
        def on_connect():
            print('connect')

        def on_disconnect():
            print('disconnect')

        def on_reconnect():
            print('reconnect')

        def on_request_update(*args):
            """
            launches the local training
            """
            req = args[0]
            self.logger.info(("update requested"))
            self.logger.info(req)
            cur_round = req['round_number']
            room_id = req['room_id']
            self.logger.info("### Round {} ###".format(cur_round))

            if cur_round == 0:
                self.logger.info("received initial model")
                print(req['current_weights'])
                weights = pickle_string_to_obj(req['current_weights'])
                self.local_model.set_weights(weights)

            self.logger.info("Before Launching The training")
            my_weights, train_loss = self.local_model.train_one_round()
            self.logger.info("After Launching The Training")

            pickle_to_file(my_weights, str(room_id) + '.pkl')
            resp = {
                'round_number': cur_round,
                'train_size': self.local_model.model.train_size,
                'train_loss': train_loss,
                'room_id': room_id
            }

            self.logger.info("client_train_loss {}".format(train_loss))

            if 'aggregation' in req and req['aggregation']:
                client_accu, client_loss = self.local_model.evaluate()
                client_accu = np.nan_to_num(client_accu)
                resp['client_test_loss'] = client_loss
                resp['client_accu'] = client_accu
                resp['client_test_size'] = self.local_model.model.valid_size
                self.logger.info("client_test_loss {}".format(client_loss))
                self.logger.info("client_accu {}".format(client_accu))

            print("Emit client_update")
            self.sio.emit('client_update', resp)
            self.logger.info("sent trained model to server")
            print("Emited...")

        def on_stop_and_eval(*args):
            """
            receives aggregated model from server and evaluate the model
            """
            self.logger.info("received aggregated model from server")
            req = args[0]
            cur_time = time.time()
            if req['weights_format'] == 'pickle':
                weights = pickle_string_to_obj(req['current_weights'])
                self.local_model.set_weights(weights)
            print('get weights')

            self.logger.info("receiving weight time is {}".format(time.time() - cur_time))
            server_accu, server_loss = self.local_model.evaluate()
            self.logger.info("after evaluating")
            server_accu = np.nan_to_num(server_accu)

            resp = {
                'test_size': self.local_model.model.valid_size,
                'test_loss': server_loss,
                'test_accuracy': float(server_accu)
            }

            print("Emit client_eval")
            self.sio.emit('client_eval', resp)

            if req['STOP']:
                print("Federated training finished ...")
                exit(0)

        def on_check_client_resource(*args):
            req = args[0]
            print("check client resource.")
            if self.ignore_load:
                load_average = 0.15
                print("Ignore load average")
            else:
                load_average = self.load_stat()
                print("Load average:", load_average)

            resp = {
                'round_number': req['round_number'],
                'load_rate': load_average
            }
            self.sio.emit('check_client_resource_done', resp)

        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)
        self.sio.on('init', self.on_init)
        self.sio.on('request_update', on_request_update)
        self.sio.on('stop_and_eval', on_stop_and_eval)
        self.sio.on('check_client_resource', on_check_client_resource)

    def intermittently_sleep(self, p=.1, low=10, high=100):
        """
        random sleep time
        """
        if (random.random() < p):
            time.sleep(random.randint(low, high))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, help="which GPU to run")
    parser.add_argument("--config_file", type=str, required=True, help="task config file")
    parser.add_argument("--ignore_load", default=True, help="wheter ignore load of not")
    parser.add_argument("--port", type=int, required=True, help="server port")
    opt = parser.parse_args()
    FederatedClient("http://localhost", opt.port, opt.config_file, opt.gpu, opt.ignore_load)
