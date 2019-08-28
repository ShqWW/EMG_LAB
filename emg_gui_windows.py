from PyQt5.QtWidgets import (QMainWindow,QApplication,QComboBox,QDesktopWidget,QPushButton,QWidget,QLabel,
                             QLineEdit,QCheckBox,QVBoxLayout,QMessageBox)
from PyQt5.QtGui import QIcon,QPalette,QColor,QPixmap
from PyQt5.QtCore import Qt,QTimer,QCoreApplication
import threading

def screen_center(window,width,height):
    window.resize(width,height)
    screen_size=QDesktopWidget().screenGeometry()
    window.move((screen_size.width()-width)/2,(screen_size.height()-height)/2)
    window.setWindowIcon(QIcon('./source/appicon.png'))
#set the window color
def set_color(window,R=245,G=222,B=179):
    palette = QPalette()
    palette.setColor(window.backgroundRole(), QColor(R,G,B))
    window.setPalette(palette)

class First_window(QMainWindow):
    def __init__(self,main_core):
        super().__init__()
        
        self.main_core=main_core
        set_color(self)
        window_width,window_height=400,600
        screen_center(self,window_width,window_height)
        self.setWindowTitle("set the IP and sensor")
        self.status=self.statusBar()
        self.status.showMessage("please set the IP and Port")
        self.status.setStyleSheet("font:15pt")
        
        self.add_label_editline()
        self.add_checkbox()
        self.add_button()
  
    def add_button(self):
        btn=QPushButton('Confirm',self)
        btn.setStyleSheet("font:15pt;background-color:rgb(72,209,204);")
        btn.clicked.connect(self.send_para)
        btn.resize(btn.sizeHint())
        btn.move(160,520)
        btn.setShortcut(Qt.Key_Return)#设置回车快捷键
    def add_label_editline(self):
        label1=QLabel(self)
        label1.setText("Person ID:")
        label1.setStyleSheet("font:15pt")
        label1.move(30,16)
        self.line1=QLineEdit(self)
        with open("./source/ID.txt","r") as f:         
            line = f.readline()
        self.line1.setText(line)
        self.line1.setStyleSheet("font:15pt")
        self.line1.resize(self.line1.sizeHint())
        self.line1.move(160,13)
        
        label2=QLabel(self)
        label2.setText("IP Address:")
        label2.setStyleSheet("font:15pt")
        label2.resize(label2.sizeHint())
        label2.move(23,76)
        self.line2=QLineEdit(self)
        with open("./source/IP.txt","r") as f:         
            line = f.readline()
        self.line2.setText(line)
        self.line2.setStyleSheet("font:15pt;")
        self.line2.resize(self.line2.sizeHint())
        self.line2.move(160,73)

        label3=QLabel(self)
        label3.setText("Please select the channel")
        label3.setStyleSheet("font:15pt")
        label3.resize(label3.sizeHint())
        label3.move(90,130)
    def add_checkbox(self):
        box_label=[]
        self.checkBox=[]
        for i in range(4):
            for j in range(4):
                self.checkBox.append(QCheckBox(self))
                self.checkBox[4*i+j].setStyleSheet("QCheckBox::indicator { width: 20px; height: 20px;}")
                self.checkBox[4*i+j].move(40+90*j,175+90*i)
                if i<2 and j<4:
                    self.checkBox[4*i+j].setChecked(True)
                box_label=QLabel(self)
                box_label.setText("s"+str(4*i+j+1))
                box_label.setStyleSheet("font:15pt")
                box_label.move(65+90*j,173+90*i)
    def send_para(self):
        ID=self.line1.text()
        IP=self.line2.text()
        with open("./source/ID.txt","w") as f:
            f.write(ID)
        with open("./source/IP.txt","w") as f:
            f.write(IP)
        channel=[]
        for i in range(16):
            if (self.checkBox[i].isChecked()==True):
                channel.append(i+1)
        self.main_core.para_pass(ID,IP,channel)
        self.close()
        self.next_window=connecting_window(self.main_core)
        self.next_window.show()
        self.next_window.connecting()
class connecting_window(QWidget):
    def __init__(self,main_core):
        self.main_core=main_core
        super().__init__()
        self.setWindowTitle("Connection")
        set_color(self)
        window_width,window_height=400,100
        screen_center(self,window_width,window_height)
        self.label=QLabel(self)
        self.label.setText("Connecting to the server")
        self.label.setStyleSheet("font:15pt")
        self.label.move(90,16)
        self.timer = QTimer(self) #初始化一个定时器
        self.ctn=0
        self.add_button()
    def connecting(self): 
        threading.Thread(target=self.main_core.connect_server).start()
        self.timer.timeout.connect(self.check) #计时结束调用operate()方法
        self.timer.start(200) #设置计时间隔并启动
    def check(self):
        if self.main_core.connecting:
            pass
        else:
            self.close()
            self.next_window=connected_window(self.main_core)
            self.next_window.change_label()
            self.timer.stop()
            self.next_window.show()
            
    def add_button(self):
        btn=QPushButton('cancel',self)
        btn.setStyleSheet("font:12pt;background-color:rgb(72,209,204);")
        btn.clicked.connect(self.cancel)
        btn.resize(btn.sizeHint())
        btn.move(165,55)
        btn.setShortcut(Qt.Key_Return)#设置回车快捷键
    def cancel(self):
        #first_window.show()
        self.close()
        #QCoreApplication.instance().quit
class connected_window(QWidget):
    def __init__(self,main_core):
        self.main_core=main_core
        super().__init__()
        self.setWindowTitle("Connection")
        set_color(self)
        window_width,window_height=400,100
        screen_center(self,window_width,window_height)
        self.label=QLabel(self)
        self.label.setStyleSheet("font:15pt")
        self.label.move(115,13)
        self.add_button()
        
    def add_button(self):
        btn=QPushButton('Confirm',self)
        btn.setStyleSheet("font:12pt;background-color:rgb(72,209,204);")
        btn.clicked.connect(self.check)
        btn.resize(btn.sizeHint())
        btn.move(165,55)
        btn.setShortcut(Qt.Key_Return)#设置回车快捷键
    def change_label(self):
        if self.main_core.con_success:
            self.label.setText("Connection succeed!")
        else:
            self.label.setText("Connection failed!")
    def check(self):
        if self.main_core.con_success:
            print("success")
            self.next_window=init_train_test_window(self.main_core)
            self.next_window.show()
            self.close()
        else:
            print("fail")
            self.next_window=init_train_test_window(self.main_core)
            self.next_window.show()
            self.close()
class init_train_test_window(QWidget):
    def __init__(self,main_core):
        super().__init__()
        self.main_core=main_core
        self.method_choose="SVM"
        self.train_num=1
        self.mode="Train"
        self.gestures=[]
        window_width,window_height=650,870
        screen_center(self,window_width,window_height)
        set_color(self)
        self.setWindowTitle("initial model")
        self.add_combox()
        self.add_label()
        self.add_picture_label_checkbox()
        self.add_button()
    def add_label(self):
        label1=QLabel(self)
        label1.setText("Model:")
        label1.setStyleSheet("font:15pt")
        label1.move(253,23)
        label2=QLabel(self)
        label2.setText("Mode:")
        label2.setStyleSheet("font:15pt")
        label2.move(259,73)
        label3=QLabel(self)
        label3.setText("Number of train sessions:")
        label3.setStyleSheet("font:15pt")
        label3.move(80,123)
    def add_combox(self):
        self.method_list=["SVM","ELM"]
        self.train_num_list=["1","2","3","4","5"]
        self.mode_list=["Train","Predict"]
        self.image_list=["relax","hold","split","cwise","ccwise","thumb","stretch","ok","6+1"]
        self.cb_method=QComboBox(self)
        self.cb_method.addItems(self.method_list)
        self.cb_method.setStyleSheet("font:15pt;")
        self.cb_method.move(350,20)
        self.cb_method.currentIndexChanged.connect(self.change_method)
        self.cb_train_test=QComboBox(self)
        self.cb_train_test.addItems(self.mode_list)
        self.cb_train_test.setStyleSheet("font:15pt;")
        self.cb_train_test.move(350,70)
        self.cb_train_test.currentIndexChanged.connect(self.change_mode)
        self.cb_train_num=QComboBox(self)
        self.cb_train_num.addItems(self.train_num_list)
        self.cb_train_num.setStyleSheet("font:15pt;")
        self.cb_train_num.move(350,120)
        self.cb_train_num.currentIndexChanged.connect(self.change_train_num)
    def add_picture_label_checkbox(self):
        label_list=[]
        picture_list=[]
        self.checkBox_list=[]
        for i in range(9):
            label_list.append(QLabel(self))
            picture_list.append(QLabel(self))
            self.checkBox_list.append(QCheckBox(self))
        for i in range(9):
            label_list[i].setText(self.image_list[i])
            label_list[i].setStyleSheet("font:15pt")
            picture_list[i].setPixmap(QPixmap("./source/"+self.image_list[i]+".jpg"))
            self.checkBox_list[i].setStyleSheet("QCheckBox::indicator { width: 20px; height: 20px;}")
            if i<5:
                self.checkBox_list[i].setChecked(True)
        for i in range(3):
            label_list[i].move(90+190*i,190)
            self.checkBox_list[i].move(60+190*i,197)
            picture_list[i].move(40+190*i,230)
        for i in range(3):
            label_list[3+i].move(90+200*i,180+190)
            self.checkBox_list[3+i].move(60+200*i,180+197)
            picture_list[3+i].move(40+200*i,230+173)
        for i in range(3):
            label_list[6+i].move(100+200*i,150+190*2)
            self.checkBox_list[6+i].move(70+200*i,150+190*2+7)
            picture_list[6+i].move(40+200*i,230+167*2)
    def add_button(self):
        btn=QPushButton('Confirm',self)
        btn.setStyleSheet("font:15pt;background-color:rgb(72,209,204);")
        btn.clicked.connect(self.init_core)
        btn.resize(btn.sizeHint())
        btn.move(300,800)
        btn.setShortcut(Qt.Key_Return)#设置回车快捷键
    def change_method(self,text):
        self.method_choose=self.method_list[text]
    def change_train_num(self,text):
        self.train_num=int(self.train_num_list[text])
    def change_mode(self,text):
        self.mode=self.mode_list[text]
    def init_core(self):
        for i in range(9):
            if (self.checkBox_list[i].isChecked()==True):
                self.gestures.append(self.image_list[i])
        self.main_core.init_train_test(self.gestures,self.method_choose,self.mode,self.train_num)
        self.close()
        if self.main_core.mode=="Train":
            self.train_gui=train_guide(self.main_core)
            self.train_gui.work()
        elif self.main_core.mode=="Predict":
            self.pre=predict_guide(self.main_core)
            self.pre.work()

        

#three training window       
class perform(QWidget):
    def __init__(self,main_core,train_gui):
        self.main_core=main_core
        self.train_gui=train_gui
        super().__init__()
        window_width,window_height=300,200
        screen_center(self,window_width,window_height)
        set_color(self)
        self.setWindowTitle("training")
        self.label=QLabel(self)
        self.picture=QLabel(self)
        self.label.setStyleSheet("font:15pt")
        self.vlayout=QVBoxLayout()
        self.timer=QTimer()
        self.timer.timeout.connect(self.showing)
    def change_label(self,some):
        self.label.setText("please hold:"+some)
        self.label.resize(self.label.sizeHint())
        self.picture.setPixmap(QPixmap("./source/"+some+".jpg"))
        self.vlayout.addWidget(self.label,0,Qt.AlignVCenter|Qt.AlignCenter)
        self.vlayout.addWidget(self.picture,0,Qt.AlignVCenter|Qt.AlignCenter)
        self.setLayout(self.vlayout)
    def start_show(self):
        self.show()
        threading.Thread(target=self.main_core.collect_train).start()
        self.timer.start(100)
    def showing(self):
        if self.main_core.collecting==False:
            self.timer.stop()
            self.close()
            self.train_gui.work()
      
class ctnd_window(QWidget):
    def __init__(self,train_gui):
        self.train_gui=train_gui
        super().__init__()
        window_width,window_height=586,339
        screen_center(self,window_width,window_height)
        self.setWindowTitle("be ready")
        self.label1=QLabel(self)
        self.label1.setPixmap(QPixmap("./source/cntd3.png"))
        self.label1.resize(self.label1.sizeHint())
        self.label1.move(0,0) 
        self.timer = QTimer(self) #初始化一个定时器
        self.timer.timeout.connect(self.ctn) #计时结束调用operate()方法
    def start_show(self):
        self.show()
        self.timer.start(1000) #设置计时间隔并启动
        self.timr=3
    def ctn(self):
        self.timr-=1
        self.label1.setPixmap(QPixmap("./source/cntd"+str(self.timr)+".png"))
        if self.timr==0:
            self.timer.stop()
            self.close()
            self.label1.setPixmap(QPixmap("./source/cntd3.png"))
            self.train_gui.work()

class cafebreak_window(QWidget):
    def __init__(self,train_gui):
        self.train_gui=train_gui
        super().__init__()
        window_width,window_height=586,339
        screen_center(self,window_width,window_height)
        picture=QLabel(self)
        picture.setPixmap(QPixmap("./source/break.png"))
        picture.move(0,0)
        self.timer=QTimer()
        self.timer.timeout.connect(self.showing)
    def set_session(self,num):
        self.setWindowTitle("Session "+str(num))
    def start_show(self):
        self.show()
        self.timer.start(1000)
        self.show_time=5
    def showing(self):
        self.show_time-=1
        if self.show_time==0:
            self.timer.stop()
            self.close()
            self.train_gui.work()
            
class train_guide():
    def __init__(self,main_core):
        self.main_core=main_core
        self.perform_win=perform(self.main_core,self)
        self.cf_win=cafebreak_window(self)
        self.ctn_win=ctnd_window(self)
        
         #初始化一个定时器
        self.display_record=0
        self.session_array=0
        self.gesture_array=0
        self.ready_array=0
    def work(self):
        self.gesture_num=len(self.main_core.gestures)
        if self.session_array < self.main_core.session_num:
            if self.display_record==0:
                self.display_record=1
                self.cf_win.set_session(self.session_array+1)
                self.cf_win.start_show()
            else:
                if self.ready_array==0:
                    self.ready_array=1
                    self.ctn_win.start_show()
                else:
                    self.ready_array=0
                    self.perform_win.change_label(self.main_core.gestures[self.gesture_array])
                    self.perform_win.start_show()
                    self.gesture_array=(self.gesture_array+1)%self.gesture_num
                    if self.gesture_array==0:
                        self.display_record=0
                        self.session_array=self.session_array+1
        else:
            self.main_core.train()
            self.predict_guide=predict_guide(self.main_core)
            self.predict_guide.work()

class predict_window(QWidget):
    def __init__(self,main_core):
        self.main_core=main_core
        super().__init__()
        window_width,window_height=300,200
        screen_center(self,window_width,window_height)
        set_color(self)
        self.setWindowTitle("Predicting")
        self.label=QLabel(self)
        self.picture=QLabel(self)
        self.label.setStyleSheet("font:15pt")
        self.vlayout=QVBoxLayout()
        
    def change_label(self,some,pro):
        self.label.setText("current geature is:"+" "+some+" "+pro)
        self.label.resize(self.label.sizeHint())
        self.picture.setPixmap(QPixmap("./source/"+some+".jpg"))
        self.vlayout.addWidget(self.label,0,Qt.AlignVCenter|Qt.AlignCenter)
        self.vlayout.addWidget(self.picture,0,Qt.AlignVCenter|Qt.AlignCenter)
        self.setLayout(self.vlayout)
    def start_show(self):
        self.show()
        self.t=threading.Thread(target=self.main_core.predict)
        self.t.start()
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QMessageBox.Yes | 
            QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.main_core.TERMINATED=True           
            event.accept()
        else:
            event.ignore()
class predict_guide():
    def __init__(self,main_core):
        self.main_core=main_core
        self.timer=QTimer()
        self.timer.timeout.connect(self.updating)
        self.window=predict_window(self.main_core)
    def work(self):
        self.window.start_show()
        self.timer.start(200)
    def updating(self):
        self.window.change_label(self.main_core.y_pre[0],self.main_core.y_pre[1])





