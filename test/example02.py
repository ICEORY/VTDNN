import tkinter as tk


class Application(object):
    def __init__(self):
        # define a window
        self.window = tk.Tk()
        self.window.title("application demo")
        # define a frame
        self.frame = tk.Frame()
        self.frame.pack()
        # create buttons
        self.v1 = tk.IntVar()
        self.cbtBold = tk.Checkbutton(
            self.frame, text="bold", variable=self.v1, command=self.processCheckButton)
        self.cbtBold.grid(row=1, column=1)
        # define another frame
        self.frame2 = tk.Frame(self.window)
        self.frame2.pack()
        # create a label
        self.label = tk.Label(self.frame2, text="type your name")
        self.name = tk.StringVar()
        self.entryName = tk.Entry(self.frame2, textvariable=self.name)
        self.btGetName = tk.Button(
            self.frame2, text="get name", command=self.processButton)
        self.label.grid(row=1, column=1)
        self.entryName.grid(row=1, column=2)
        self.btGetName.grid(row=1, column=3)
        # create text
        self.text = tk.Text(self.window)
        self.text.pack()
        self.text.insert(tk.END, "Tips:\n do some thing")
        self.window.mainloop()

    def processCheckButton(self):
        print("state of check button:" +
              ("selected" if self.v1.get() == 1 else "unselected"))

    def processButton(self):
        print("your name is: "+self.name.get())


def main():
    demo = Application()


if __name__ == '__main__':
    main()
