import sys
import platform
import emg_core
if platform.platform()[0]=="L":
    import emg_gui_linux as emg_gui
elif platform.platform()[0]=="W":
    import emg_gui_linux as emg_gui


if __name__=="__main__":
    c=emg_core.core()
    app=emg_gui.QApplication(sys.argv)
    train_gui=emg_gui.train_guide(c)
    first_window=emg_gui.First_window(c,train_gui)
    first_window.show()
    sys.exit(app.exec_())

