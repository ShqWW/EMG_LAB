import sys
import platform
import emg_core
if platform.platform()[0]=="L":
    import emg_gui_linux as emg_gui
elif platform.platform()[0]=="W":
    import emg_gui_linux as emg_gui


if __name__=="__main__": 
    app=emg_gui.QApplication(sys.argv)
    main_window=emg_gui.First_window(emg_core.core())
    main_window.show()
    sys.exit(app.exec_())

