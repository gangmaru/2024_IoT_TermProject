## Human Activity Recognition with Wi-Fi Sensing
A Wi-Fi sensing program that converts and visualizes a pcap file that recognizes five actions of a user into csv, communicates in real time on a server and a client, and recognizes five actions of a user.
As a tool for collecting Wi-Fi csi, firmware is installed on Raspberry Pi 3 to collect csi. 
The collected csi is transmitted to the server through socket communication, and the server preprocesses the csi data at a predefined window size and uses it as an input value of the trained activity classification model.

We can recognize the behavior below by analyzing it.
- empty
- stand
- walk
- run
- sleep

#### ※ Notice ※
Raspberry Pi 3: Client sending csi data that has completed nexmon firmware installation
Raspberry Pi 4: Activate AP mode
User pc: Server communicating with the client. Communicating with Raspberry Pi 3 through an socket communication.

### Open Source Usage
#### 1. [pcap-to-csv](https://github.com/cheeseBG/pcap-to-csv)
Provides Python scripts for converting packet capture files (PCAPs) collected using Nexmon firmware into CSV files

#### 2. [csi-visualization](https://github.com/cheeseBG/csi-visualization)
Tools for visualizing Wi-Fi Channel State Information (CSI) data

#### 3. [mowa-wifi-sensing](https://github.com/oss-inc/mowa-wifi-sensing)
It performs real-time behavior recognition of csi data collected by wifi sensing.

### How to use
1. Enter the csv file you collected in the `domain_A` in the `csi_dataset`

2. Create `server/checkpoint` folder.

3. Modify `config.yaml` to match server and client configurations(server ip, client mac)

4. `run_SVL.py` encountered an execution error in the corresponding repository, adding the following changes

4-1. Changed use_cuda to check via torch.cuda.is _available().

```
use_cuda = config['GPU']['cuda'] and torch.cuda.is_available()
```

4-2. In torch.load, we added the map_location parameter so that it can be mapped to the CPU according to CUDA availability.

```
model.load_state_dict(torch.load(config['application']['SVL']['save_model_path'], map_location='cuda' if use_cuda else 'cpu'))
```

4-3. We used torch.device ('cuda') when we moved the model to CUDA.

```
if use_cuda:
    model.to(torch.device('cuda'))
```

```
if use_cuda:
    c_data = c_data.cuda()
```


This modified script does not cause problems when CUDA is not available, and if CUDA is available, the GPU will be used automatically.
