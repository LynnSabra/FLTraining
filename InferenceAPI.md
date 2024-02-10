
## Inference API
Similar to the inference script, An inference API is served to test your trained models.

### Build the Inference API Image
The inference API Image should be built outside the FL benchmark container, taking into consideration that the FL benchmark image is already built

```sh
cd inference_api
sudo docker build -t fl_inference .
```

### Run the Inference API Container

To run the Inference API, go to the project's directory and run the following:

#### On GPU

```shell
sudo docker run --runtime=nvidia -v $(pwd)/dataset:/dataset -itp 5555:5555 fl_inference
```

#### On CPU

```shell
sudo docker run -e USE_CPU='TRUE' -v $(pwd)/dataset:/dataset -itp 5555:5555 fl_inference
```

The API will now listen on port 5555. For a simple test, run the APIclient script

```
python3 client.py
