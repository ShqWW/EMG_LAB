import numpy as np
import time
from socket import*
import os
import struct
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from time import sleep
import util
import pickle
import scipy.io as scio
PATH_PREFIX = "./models"




class core():
    def __init__(self):
        self.host = ""
        self.main_port = 50040
        self.emg_port = 50043
        self.sample_unit = 4 * 16  # 4 bytes per channel, non-changeable
        self.sample_rate = 2000  # Default in DELSYS' server, non-changeable

        # Parameters
        self.sensors = []  # support multiple input channels
        self.model = "SVM"
        self.num_feat = None  # number of features
        self.sample_time = 0.5  # second. a fixed time length for processing EMG data
        self.training_unit = int(5 / self.sample_time)  # observations for one class, e.g. 5s
        self.session_num = 5
        self.gestures = [0,1,2,3,4]
        self.mode=0
        self.CTL_CLI = None
        self.EMG_CLI = None
        self.person_id=None
        self.connecting=False# 是否正在连接
        self.con_success=False#是否连接成功
        self.mode=0
        self.directory=""
        self.emg_data=None
        self.training_data = None  # NUM FEATS numbers and label
        self.train_data_tmp = None
        self.label_encoder = LabelEncoder()
        self.scaler=None
        self.accuracy=0
        self.collecting=False
        self.encoder_msg = ""
        self.TERMINATED = False
        self.CHANGE_FLAG=False
        self.y_pre=[" "," "]
        self.signal=None
    def para_pass(self,ID,IP,channel):
        self.person_id=ID
        self.host=IP
        self.sensors=channel
        print(self.sensors)
    def receive(self, sample_num=None):
        if sample_num==None:
            sample_num=int(self.sample_time * self.sample_rate)
        """
        receive raw data for one observation
        :param sample_num:
        :return: raw_emg_data, rows = samples, columns = active sensors
        """
        raw_emg_data = np.empty(shape=(sample_num, 1), dtype=object)
        # rows = samples, columns = active sensors
        for sample_index in range(sample_num):
            tmp_data_rcv = self.EMG_CLI.recv(64)
            while len(tmp_data_rcv) < 64:
                tmp_data_rcv += self.EMG_CLI.recv(1)
            raw_emg_data[sample_index] = tmp_data_rcv
        return raw_emg_data

    def connect_server(self):
        main_conf = (self.host, self.main_port)
        emg_conf = (self.host, self.emg_port)
        self.CTL_CLI = socket(AF_INET, SOCK_STREAM)
        self.EMG_CLI = socket(AF_INET, SOCK_STREAM)
        self.CTL_CLI.settimeout(5)
        self.EMG_CLI.settimeout(5)#timeout limited 
        # Connect to DELSYS' server
        self.connecting=True
        try:
            self.CTL_CLI.connect(main_conf)
            self.EMG_CLI.connect(emg_conf)
            self.con_success=True
        except:
            self.con_success = False  
        finally:
            self.CTL_CLI.settimeout(None)
            self.EMG_CLI.settimeout(None)#timeout unlimited
            self.connecting = False
        # self.connecting=True
        # time.sleep(3)
        # self.con_success=True
        # self.connecting=False
    def init_train_test(self,gestures,model,mode,train_num):
        self.gestures=gestures
        self.model=model
        self.mode=mode
        self.session_num=train_num
        #  self.directory="./"+self.person_id+"/"
        #if not os.path.exists(self.directory):
        #    os.makedirs(self.directory)
        self.num_feat = 6 * len(self.sensors)
        self.training_data = np.zeros(shape=(1, self.num_feat + 1))
        self.train_data_tmp = np.ones(shape=(1, self.num_feat + 1))
        self.emg_data = np.zeros((1, len(self.sensors)))
        self.signal = np.zeros(shape=(int(self.sample_time* self.sample_rate), 8))
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.gestures)
        for gesture in self.gestures:
            self.encoder_msg += "{}\timplies\t{}\n".format(
                gesture, self.label_encoder.transform([gesture])
            )
    def collect_train(self):
        self.collecting=True
        ctl_data = "START\r\n\r\n"
        self.CTL_CLI.send(ctl_data.encode())

        for _ in range(self.training_unit):
            # receive one observation
            raw_emg_data = self.receive()
            # decode raw data into floats
            emg_data = np.zeros(shape=(int(self.sample_time * self.sample_rate), len(self.sensors)))

            for (sample_index, _), chunk_data in np.ndenumerate(raw_emg_data):
                for id_index, sensor_id in enumerate(self.sensors):
                    raw_tmp = chunk_data[(sensor_id - 1) * 4 : sensor_id * 4]
                    decoded_data = struct.unpack("f", raw_tmp)[0]
                    emg_data[sample_index][id_index] = decoded_data

            # # fake data
            #sleep(self.sample_time)
            #emg_data = np.random.rand(int(self.sample_time * self.sample_rate), len(self.sensors))

            # update recorded emg
            self.emg_data = np.append(self.emg_data, emg_data, axis=0)

            # calculate feature
            self.train_data_tmp[0, 1:] = util.feat_gen(
                emg_data, waveLength=1, mav=1, ar4c=1
            )
            # stack training data (features)
            self.training_data = np.append(
                self.training_data, self.train_data_tmp, axis=0
            )

        # Stop transmission
        ctl_data = "STOP\r\n\r\n"
        self.CTL_CLI.send(ctl_data.encode())
        self.collecting=False
    def train(self):
        # collect training data
        cls_num = len(self.gestures)
        array=0
        for n in range(int(cls_num * self.session_num)):
            label = self.gestures[n % cls_num]
            for k in range(self.training_unit):
                self.training_data[1+array][0] = self.label_encoder.transform([label])[0]
                array+=1
        # start training
        X = self.training_data[1:, 1:]  # the first row is fake data
        Y = self.training_data[1:, 0].reshape(-1, 1)  # the first row is fake data
        self.scaler = StandardScaler().fit(X)  # normalize X
        X = self.scaler.transform(X)

        self.clf, tr_acc = util.fit(X, Y, model=self.model)

        #gui.AlertUI(msg="Training accuracy: {}".format(tr_acc))
        self.accuracy="Training accuracy: {}".format(tr_acc)
        print(self.accuracy)
        if self.model == "SVM":
            model_dict = {
                "emg": self.emg_data[1:, :],
                "X": X,
                "Y": Y,
                "scaler": self.scaler,
                "label_enc": self.label_encoder,
                "clf": self.clf,
            }

        path = "{}/model-{}.model".format(
            PATH_PREFIX, self.person_id)
        f = open(path, "wb+")
        f.write(pickle.dumps(model_dict))
        #scio.savemat(path,model_dict)

    def predict(self):
        path = "{}/model-{}.model".format(
            PATH_PREFIX, self.person_id)
        f=open(path,'rb').read()
        model_dict=pickle.loads(f)
        self.scaler=model_dict["scaler"]
        self.label_encoder=model_dict["label_enc"]
        self.clf=model_dict["clf"]


        self.data_tmp = np.ones(shape=(1, self.num_feat))

        # Init transmission
        ctl_data = "START\r\n\r\n"
        self.CTL_CLI.send(ctl_data.encode())

        #while not self.TERMINATED:
        while not self.TERMINATED:
            # receive one observation
            raw_emg_data = self.receive()

            # decode raw data into floats
            emg_data = np.zeros(shape=(int(self.sample_time * self.sample_rate), len(self.sensors)))
            for (sample_index, _), chunk_data in np.ndenumerate(raw_emg_data):
                for id_index, sensor_id in enumerate(self.sensors):
                    raw_tmp = chunk_data[(sensor_id - 1) * 4 : sensor_id * 4]
                    decoded_data = struct.unpack("f", raw_tmp)[0]
                    emg_data[sample_index][id_index] = decoded_data

            # fake data
            #sleep(self.sample_time)
            #emg_data = np.random.rand(int(self.sample_time * self.sample_rate), len(self.sensors))

            #for sensor_id, sensor in enumerate(self.sensors):
            #    self.signal[:, sensor - 1] = emg_data[:, sensor_id]

            # calculate feature
            data_tmp = util.feat_gen(emg_data, waveLength=1, mav=1, ar4c=1)
            data_tmp = self.scaler.transform(data_tmp)

            # classifying
            prediction = util.predict(data_tmp, self.clf, model=self.model)
            # self.y_pre = (
            #     self.label_encoder.inverse_transform([int(prediction[0])])[0]
            #     + str(prediction[1]),
            #     int(prediction[0]),
            #     prediction[1],
            # ) 
            print(prediction)
            self.y_pre=[self.label_encoder.inverse_transform([int(prediction[0])])[0],str(round(prediction[1],4))]
            self.CHANGE_FLAG = True

        


    


