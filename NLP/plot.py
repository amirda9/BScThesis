# importing tkinter gui
import tkinter as tk




def setup_gui(sentence):
    #creating window
    window=tk.Tk()

    #getting screen width and height of display
    width= window.winfo_screenwidth()
    height= window.winfo_screenheight()
    #setting tkinter window size
    window.geometry("%dx%d" % (width, height))
    window.title("Plot")
    ep = 0.05
    label = []
    place = 900
    i = 0
    for text in sentence:
        if text == '<PAD>':
            pass
        else:
            label.append(tk.Label(window, text=text))
            y = ((50+(int(i/10)*70)) / height)
            x = ((50+(int(i%10)*125)) / width)
            # print(x*1920,y*1080)
            label[-1].place(relx=x, rely=y)
            # label[-1].config(font=("Courier", 14))
            # label[-1].pack()
            # label.pack()
            place += 50
            i+=1
            # print(i)
    window.mainloop()
