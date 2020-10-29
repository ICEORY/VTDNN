import math
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog

import numpy as np
import torch
from PIL import Image, ImageTk

import cv2
import img_process
from deep_cnn import DeepModel
# Visualization tools for deep neural networks (VTDNN)

class DataCache(object):
    def __init__(self, data=None):
        self.data = data
        self.update_state = False

    def writeData(self, data):
        self.update_state = True
        self.data = data

    def readData(self):
        self.update_state = False
        return self.data

    def needRefresh(self):
        self.update_state = True


class Pane(object):
    def __init__(self, master, height=300, width=300, bg="White"):
        self.height = height
        self.width = width
        self.canvas = tk.Canvas(
            master, width=width, height=height, bg=bg)

        self.item_handle = {}

    def grid(self, **kw):
        self.canvas.grid(**kw)

    def setItem(self, name, item):
        self.item_handle[name] = item

    def updateItem(self, name, **kw):
        if name in self.item_handle.keys():
            self.canvas.itemconfig(self.item_handle[name], **kw)
            return True
        else:
            return False

    def updateCoords(self, name, position):
        if name in self.item_handle.keys():
            self.canvas.coords(self.item_handle[name], position)
            return True
        else:
            return False


class Application(tk.Tk):
    """
    GUI controller
    """

    def __init__(self):
        super(Application, self).__init__()
        # ------------------------ init containers ----------------------------------
        self.debug_flag = False
        # self.iconbitmap(default="visual_tool.ico")
        self.title("VTDNN")
        # self.geometry("1510x950")
        self.resizable(width=False, height=False)

        # create container
        self.frame = tk.Frame(self)
        self.frame.pack(side="top", fill="both", expand=True)
        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(0, weight=1)

        # define deep model
        self.deep_model = DeepModel()

        # ------------------------ Menu Bar ------------------------------------------
        # define menu bar
        self.menubar = tk.Menu(self)
        self.config(menu=self.menubar)

        # define main menu
        self.main_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Menu", menu=self.main_menu)

        self.main_menu.add_command(
            label="Load Image", command=self.btnLoadImage)

        self.main_menu.add_command(
            label="From Camera", command=self.btnFromCamera)

        self.model_menu = tk.Menu(self.main_menu, tearoff=0)
        self.selected_model = tk.StringVar()
        self.selected_model.set(self.deep_model.model_list[0])
        for model_name in self.deep_model.model_list:
            self.model_menu.add_radiobutton(label=model_name,
                                            variable=self.selected_model,
                                            command=self.btnSelectModel)

        self.main_menu.add_cascade(label="Select Model", menu=self.model_menu)

        self.main_menu.add_command(
            label="Load Weight", command=self.btnLoadWeight)

        # define layer menu
        self.selected_layer = tk.StringVar(self.menubar)
        self.layer_menu = tk.Menu(self.menubar, tearoff=0)
        self.selected_layer.set(
            list(self.deep_model.layer_feature.keys())[0])
        for key in self.deep_model.layer_feature.keys():
            self.layer_menu.add_radiobutton(label=key,
                                            variable=self.selected_layer,
                                            command=self.btnSelectLayer)

        self.menubar.add_cascade(label="Layer", menu=self.layer_menu)

        # define help menu
        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_command(label="About", command=self.aboutPage)

        # ---------------------------- event --------------------------------------
        self.protocol('WM_DELETE_WINDOW', self.closeWindow)

        # ---------------------------- init canvas ---------------------------------
        self.input_canvas = Pane(
            self.frame, width=300, height=300)

        self.feature_canvas = Pane(
            self.frame, width=900, height=900)
        self.feature_canvas.canvas.bind("<Button-1>", self.btnMouseClick)

        self.scale_canvas = Pane(
            self.frame, width=300, height=300)

        self.dconv_canvas = Pane(
            self.frame, width=300, height=300)

        self.information_canvas = Pane(
            self.frame, width=300, height=900)

        # set layout
        self.input_canvas.grid(row=0, column=0, sticky="nw")
        self.feature_canvas.grid(
            row=0, column=1, rowspan=3, columnspan=3, sticky="nw")
        self.scale_canvas.grid(row=1, column=0, sticky="nw")
        self.dconv_canvas.grid(row=2, column=0, sticky="nw")
        self.information_canvas.grid(row=0, column=4, rowspan=3, sticky="nw")

        # -------------------------- message lable -----------------------------
        # define message label
        self.msg_label = tk.Label(
            self.frame, text="Message: Thanks for using VTDNN")
        self.msg_label.grid(row=3, column=0, columnspan=3, sticky="nw")

        # -------------------------- init data ----------------------------------
        self.input_data = DataCache()
        self.feature_data = DataCache()
        self.scale_feature_data = DataCache()
        self.information_data = DataCache()
        self.about_page_icon = None
        self.global_icon = None

        self.selected_channel = 0
        self.model_input = None

        self.rect_params = {
            "x_0": 0,
            "y_0": 0,
            "x_1": 10,
            "y_1": 10,
            "line_width": 2,
            "color": "Red",
        }

        self.img_path = "./src/mnist_test.jpg"
        self.from_camera_state = False  # if true: read img from camera
        self.windows_state = True
        print("Application initialized done")
        self.run()

    # ---------------------------------- main function ----------------------------
    def run(self):
        self.iconbitmap('./src/vtcnn_16x16.ico')
        # ------------------------- init processing -------------------------------
        self.updateImagePath(self.img_path)
        self.updateInformation()
        main_thread = threading.Thread(target=self.updateCanvasThread)
        main_thread.start()

    # --------------------------- trigger event ------------------------------------
    def closeWindow(self):
        print("exit application")
        self.windows_state = False
        if self.from_camera_state:
            self.from_camera_state = False
            time.sleep(0.5)

        self.destroy()
    def btnSelectModel(self):
        self.deep_model = DeepModel(net_type=self.selected_model.get())
        self.layer_menu = tk.Menu(self.menubar, tearoff=0)
        self.selected_layer.set(
            list(self.deep_model.layer_feature.keys())[0])
        for key in self.deep_model.layer_feature.keys():
            btn = self.layer_menu.add_radiobutton(label=key,
                                            variable=self.selected_layer,
                                            command=self.btnSelectLayer)

        self.menubar.entryconfigure(index=2, menu=self.layer_menu)
        self.input_data.needRefresh()
        self.updateInformation()

    def btnSelectLayer(self):
        self.feature_data.needRefresh()
        
    def updateFeature(self):
        layer = self.selected_layer.get()
        if self.selected_channel >= self.deep_model.layer_feature[layer].size(1):
            self.selected_channel = 0
        if layer in self.deep_model.layer_feature.keys():
            # start_t = time.time()
            if "Pr" in layer:
                print(torch.nonzero(self.deep_model.layer_mask[layer]==0).squeeze())
            grid_img = img_process.tensor2Grid(
                self.deep_model.layer_feature[layer])
            # print("time: ", time.time()-start_t)
            img_tk, _ = img_process.loadFromNumpy(
                grid_img,
                img_width=self.feature_canvas.width,
                img_height=self.feature_canvas.height)
            if img_tk is not None:
                self.feature_data.writeData(img_tk)

    def btnLoadImage(self):
        self.from_camera_state = False
        path = filedialog.askopenfilename(
            initialdir="./src/",
            title="Select image file",
            filetypes=(("Image file", "*.jpg;*.JPEG;*.bmp;*.PNG"),
                       ("All types", "*.*"))
        )
        self.updateImagePath(path)

    def updateImagePath(self, path):
        if path is not None and os.path.isfile(path):
            img_tk, self.model_input = img_process.loadFromFile(
                path=path,
                img_width=self.input_canvas.width,
                img_height=self.input_canvas.height
            )
            if img_tk is not None:
                self.input_data.writeData(img_tk)
                self.img_path = path

    def btnLoadWeight(self):
        path = filedialog.askopenfilename(
            initialdir="./pretrained/",
            title="Select weight file",
            filetypes=(("Weight file", "*.pkl;*.pth"),
                       ("All types", "*.*"))
        )
        if path is not None and os.path.isfile(path):
            self.deep_model.loadWeight(path)
            self.input_data.needRefresh()

    def btnFromCamera(self):
        print("click camera")
        self.from_camera_state = True
        camera_thread = threading.Thread(target=self.cameraThread)
        camera_thread.start()

    def btnMouseClick(self, event):
        feature = self.deep_model.layer_feature[self.selected_layer.get()]
        channels = feature.size(1)
        cols = int(math.sqrt(channels))
        rows = math.ceil(channels / cols)
        unit_height = self.feature_canvas.height / rows
        unit_width = self.feature_canvas.width / cols
        index_row = int(event.y / unit_height)
        index_col = int(event.x / unit_width)
        selected_channel = index_row * cols + index_col
        if selected_channel < channels:
            self.selected_channel = selected_channel
        self.feature_data.needRefresh()

    def updateRectangle(self):
        feature = self.deep_model.layer_feature[self.selected_layer.get()]
        channels = feature.size(1)
        cols = int(math.sqrt(channels))
        rows = math.ceil(channels / cols)
        unit_height = self.feature_canvas.height / rows
        unit_width = self.feature_canvas.width / cols

        index_row = math.floor(self.selected_channel / cols)
        index_col = self.selected_channel % cols

        center_x = (index_col+0.5)*unit_width
        center_y = (index_row+0.5)*unit_height
        rect_w = unit_width * 0.95
        rect_h = unit_height * 0.95
        self.rect_params["x_0"] = center_x-rect_w/2
        self.rect_params["y_0"] = center_y-rect_h/2
        self.rect_params["x_1"] = center_x+rect_w/2
        self.rect_params["y_1"] = center_y+rect_h/2

    # ---------------------------- Thread function --------------------------------
    def cameraThread(self):
        try:
            cap = cv2.VideoCapture(0)
            while self.from_camera_state and cap.isOpened():
                # read frame
                print("enter camera")
                ret, frame = cap.read()
                if ret == True and not self.feature_data.update_state:
                    frame = frame[..., ::-1]  # bgr to rgb
                    img_tk, img = img_process.loadFromNumpy(
                        frame,
                        img_height=self.input_canvas.height,
                        img_width=self.input_canvas.width,
                        mode=Image.ANTIALIAS,
                        scale=False,
                        cmap=False)

                    if img_tk is not None:
                        self.input_data.writeData(img_tk)
                        self.model_input = img

                if not self.from_camera_state:
                    break
            print("release camera")
            cap.release()
            cv2.destroyAllWindows()
        except:
            print("failed to read camera")
        finally:
            self.from_camera_state = False

    def updateCanvasThread(self):
        # update input_canvas
        while self.windows_state:
            if self.input_data.update_state:
                print("update input canvas")
                flag = self.input_canvas.updateItem(
                    name="input_img", image=self.input_data.readData())
                if not flag:
                    img_handle = self.input_canvas.canvas.create_image(
                        0, 0, anchor="nw", image=self.input_data.readData()
                    )
                    self.input_canvas.setItem(
                        name="input_img", item=img_handle)
                # self.input_canvas.
                obr0 = self.input_data.readData()
                msg_str = self.deep_model.forward(self.model_input)
                flag = self.input_canvas.updateItem(
                    name="prediction", text=msg_str
                )
                if not flag:
                    text_handle = self.input_canvas.canvas.create_text(
                        10, 10, anchor="nw",
                        text=msg_str,
                        fill="Red",
                        font=(None, 12))
                    self.input_canvas.setItem(
                        name="prediction", item=text_handle)

                self.feature_data.needRefresh()

            # update feature_canvas
            if self.feature_data.update_state:
                print("update feature canvas")
                self.updateFeature()
                flag = self.feature_canvas.updateItem(
                    name="feature", image=self.feature_data.readData())
                if not flag:
                    img_handle = self.feature_canvas.canvas.create_image(
                        0, 0, anchor="nw", image=self.feature_data.readData()
                    )
                    self.feature_canvas.setItem(
                        name="feature", item=img_handle
                    )
                obr1 = self.feature_data.readData()  # solve blink issue

                self.updateRectangle()
                flag = self.feature_canvas.updateCoords(
                    name="hightlight_rect",
                    position=(self.rect_params["x_0"], self.rect_params["y_0"],
                              self.rect_params["x_1"], self.rect_params["y_1"])
                )

                if not flag:
                    print("create new rectangle")
                    rect_handle = self.feature_canvas.canvas.create_rectangle(
                        self.rect_params["x_0"], self.rect_params["y_0"],
                        self.rect_params["x_1"], self.rect_params["y_1"],
                        width=self.rect_params["line_width"],
                        outline=self.rect_params["color"]
                    )
                    self.feature_canvas.setItem(
                        name="hightlight_rect", item=rect_handle
                    )
                self.updateScaleFeature()

            # update scale_canvas
            if self.scale_feature_data.update_state:
                print("update scale canvas")
                flag = self.scale_canvas.updateItem(
                    name="scale_feature", image=self.scale_feature_data.readData())
                if not flag:
                    img_handle = self.scale_canvas.canvas.create_image(
                        0, 0, anchor="nw", image=self.scale_feature_data.readData()
                    )
                    self.scale_canvas.setItem(
                        name="scale_feature", item=img_handle)
                obr2 = self.scale_feature_data.readData()
                layer = self.selected_layer.get()
                feature = self.deep_model.layer_feature[layer]
                selected_feature = feature[:, self.selected_channel]
                print("dim:", selected_feature.dim())
                if selected_feature.dim() == 1:
                    msg_str = "Channel: %d\n Prop %0.4f" % (
                        self.selected_channel,
                        selected_feature.mean())
                else:
                    msg_str = "Channel: %d" % (
                        self.selected_channel)

                flag = self.scale_canvas.updateItem(
                    name="message", text=msg_str
                )
                if not flag:
                    text_handle = self.scale_canvas.canvas.create_text(
                        10, 10,
                        anchor="nw",
                        text=msg_str,
                        fill="Red",
                        font=(None, 12)
                    )
                    self.scale_canvas.setItem(name="message", item=text_handle)

            if self.information_data.update_state:
                flag = self.information_canvas.updateItem(
                    name="net_img", image=self.information_data.readData()
                )
                if not flag:
                    img_handle = self.information_canvas.canvas.create_image(
                        0, 0, anchor="nw", image=self.information_data.readData()
                    )
                    self.information_canvas.setItem(
                        name="net_img", item=img_handle)

    # ------------------------- auxiliary function --------------------------------
    def updateInformation(self):
        path = self.deep_model.visualNet()
        img_tk, _ = img_process.loadFromFile(
            path=path,
            img_width=self.information_canvas.width,
            img_height=self.information_canvas.height
        )
        if img_tk is not None:
            self.information_data.writeData(img_tk)

    def updateScaleFeature(self):
        layer = self.selected_layer.get()
        feature = self.deep_model.layer_feature[layer]
        selected_feature = feature[:, self.selected_channel].clone()
        if selected_feature.dim() == 1:
            selected_feature = selected_feature.unsqueeze(0).unsqueeze(0)
        elif selected_feature.dim() == 3:
            min_val = selected_feature.min()
            max_val = selected_feature.max()
            selected_feature.add_(-min_val).div_(max_val-min_val+1e-5)

        if selected_feature.dim() == 3 and selected_feature.size(0) == 1:
            selected_feature = torch.cat(
                (selected_feature, selected_feature, selected_feature), 0)

        img_np = selected_feature.cpu().detach().numpy(
        ) if selected_feature.is_cuda else selected_feature.detach().numpy()

        img_tk, _ = img_process.loadFromNumpy(img_np.transpose(
            (1, 2, 0)), img_width=self.scale_canvas.width, img_height=self.scale_canvas.height)

        if img_tk is not None:
            self.scale_feature_data.writeData(img_tk)

    # --------------------------- new page ---------------------------------------
    def aboutPage(self):
        window = tk.Toplevel(self)
        window.title("About VTDNN")
        window.resizable(width=False, height=False)
        window.geometry("420x240")
        window.iconbitmap("./src/vtcnn_16x16.ico")

        self.about_page_icon = tk.PhotoImage(file="./src/ico.gif")
        self.about_page_icon = self.about_page_icon.subsample(
            int(self.about_page_icon.width()/50),
            int(self.about_page_icon.height()/50)
        )
        tk.Label(window, image=self.about_page_icon).place(x=20, y=20)

        tk.Label(window, text="Visualization Tool for\nDeep Neural Networks",
                 font=(None, 20), fg="Blue", justify=tk.LEFT).place(x=80, y=20)

        tk.Label(window, text="Version: 0.0.1",
                 font=(None, 14)).place(x=20, y=100)
        tk.Label(window, text="Author: ICEORY",
                 font=(None, 14)).place(x=20, y=130)
        tk.Label(window, text="Date: 2018.05.04",
                 font=(None, 14)).place(x=20, y=160)
        tk.Label(window, text="Email: z.zhuangwei@mail.scut.edu.cn",
                 font=(None, 14)).place(x=20, y=190)
        print("about")


def main():
    app = Application()
    app.mainloop()


if __name__ == '__main__':
    main()
