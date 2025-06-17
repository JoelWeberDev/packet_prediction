# MQTT packet parsing and prediction

**Our objective is to predict to a high degree of accuracy the next packet in MQTT conversations. The MQTT is well suited for our synchronized preditive model architecture since it is commonly used, has limited band width and packets have a sender and desitination.**

## TODO
    - Convert all the data from PCAP files to CSV
        - Extract header data
        - Parse out payload data
    - Outline which features of the header can be compressed and which are fixed.
    - The predicitons will run based on the past historical data and the actual inference will run sequentially byte by byte. Any meta data that has to be transmitted raw will first be inserted and then the inference will go through until it predicts a termination as the most likely outcome. 
    - The training data should label all the raw transmission features, label all the tcp metadata. The varaible header and the payload will be left in their raw byte level format, but also parsed into the respective categories for our edification and understanding.

## Data parsing and cleaning
    - Split data into single conversations.
    - Filter the data into the categories of irrelevant, meta data, packet features

## Simulations to do
    - Payload similarity between conversations
    - Auto correlation of packet lengths plot
    - Compare packet length and similarity
    - Plot of sensor data observations
    - Get a plot of the packet deltas between each time

## Dimensions and embeddings
    - Recall that the objective is to predict the next byte in the packet sequence given a historica context of packet payloads and metadata. Since we are operating at the byte level we will have a volcabulary size of of 256.
    - This LSTM model will be multimodal meaning that along with the desired output, bytes it will allows for additional inputs such as:
        - frame.number (1 dim)
        - frame.time_delta (1 dim)
        - Past compresstion ratio **(future addition)** (1 dim)
        - mqtt.hdrcmd (dim 2^4 = 16)
        - mqtt.hdrflgs (dim 4)
        - mqtt.len (1 dim)
        - flow.direction (broker -> client = 0, client -> broker = 1) (1 dim)
        -  
        
## Model training
    - We are using an LSTM model because the data in a conversation is highly recurrent. Hence we need "memeory" of the past. Recurrent neural networks establish a running history of past data observed having an effect of autocorrelation. 
    - In the first iteration of the training we will use the single byte header imperative / flags along with the msg length to predict the content of the payload. 
    - This historical data will be the communication direction (broker to client), absolute time stamps, time deltas, the frame number, past headers, flags, message lengths, and of course the past data.
    - We will predict the data in a byte by byte format rather than attempting to do everything in a single shot.
    - Since we are operating at the byte by byte level we can limit the volcabulary to a mere 256 different possiblities. This makes the processing and size of the model very tractable.
    - For each different conversation we will need to use and store a different activation.
