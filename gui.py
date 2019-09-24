import tkinter as tk
import time
import numpy as np
from PIL import Image,ImageTk

class Data:
    def __init__(self):
        self.length = 150
        self.data = np.random.rand(14,self.length)
        self.data = self.data*200 + 4100
        self.classifier = np.random.randint(0,5)


class GUI:
    def __init__(self,parent):
        
        self.Top = parent
        self.Top.config(bg = "black")
        Top.geometry("1200x700")
        
        ##  BrainWave display
        
        self.frameWave = tk.Frame(self.Top,height = 700,width = 600,bg = 'yellow',bd = 0)
        self.frameWave.pack(side="left",fill="both",expand=1)       
        self.Refresh = tk.Button(self.frameWave, text = "Refresh", command = self.resetCanvas);
        self.Refresh.pack(fill = 'both')   
            
            ## Display Canvas (After the refresh button)
            
        self.frameEEG = tk.Frame(self.frameWave,height = 650,width = 600,bg = 'yellow',bd = 0)
        self.frameEEG.pack(fill = 'both',expand=1,side='bottom')
        
            ## Header Canvas 
        
        self.Header = tk.Canvas(self.frameEEG,height = 50,width = 600,bd=0)
        self.Header.create_text(300,30,text='EEG Data ')
        self.Header.update_idletasks()
        self.Header.pack(fill='x')
        
            ## Plot Canvas
        
        self.C = tk.Canvas(self.frameEEG,bg = "white", height = 600, width = 600,bd=0)
        self.C.pack(fill = 'x',expand=1,side='bottom')
        self.points = 150
        self.height= 20
        self.data = Data()

        ## Right frame 
        
        self.frameAction = tk.Frame(self.Top,height = 700,width = 600,bg = 'red',bd = 0)
        self.frameAction.pack(side="right",fill='both',expand=1)
        
        ## Top canvas in right frame
        
        self.Ctop = tk.Canvas(self.frameAction,bg = "black", height = 350, width = 600,bd=0)
        self.Ctop.pack(fill = 'x',side="top")
        
        ##  Bottom canvas in right frame
   
        self.Cbottom = tk.Canvas(self.frameAction,bg = "black", height = 350, width = 600,bd=0)
        self.Cbottom.pack(side = "bottom",fill = "x")
        #self.picbottom()            

        #Resize image
        
        basewidth = 300

        #Open Images
        
        self.img_t1 = Image.open("t1.png")
        wpercent = (basewidth/float(self.img_t1.size[0]))
        hsize = int((float(self.img_t1.size[1])*float(wpercent)))
        self.img_t1 = self.img_t1.resize((basewidth,hsize), Image.ANTIALIAS)
        self.img_t1 = ImageTk.PhotoImage(self.img_t1)
        
        self.img_t2 = Image.open("t2.png")
        wpercent = (basewidth/float(self.img_t2.size[0]))
        hsize = int((float(self.img_t2.size[1])*float(wpercent)))
        self.img_t2 = self.img_t2.resize((basewidth,hsize), Image.ANTIALIAS)
        self.img_t2 = ImageTk.PhotoImage(self.img_t2)
        
        self.img_t3 = Image.open("t3.png")
        wpercent = (basewidth/float(self.img_t3.size[0]))
        hsize = int((float(self.img_t3.size[1])*float(wpercent)))
        self.img_t3 = self.img_t3.resize((basewidth,hsize), Image.ANTIALIAS)
        self.img_t3 = ImageTk.PhotoImage(self.img_t3)
        
        self.img_t4 = Image.open("t4.png")
        wpercent = (basewidth/float(self.img_t4.size[0]))
        hsize = int((float(self.img_t4.size[1])*float(wpercent)))
        self.img_t4 = self.img_t4.resize((basewidth,hsize), Image.ANTIALIAS)
        self.img_t4 = ImageTk.PhotoImage(self.img_t4)

        self.img_t5 = Image.open("t5.png")
        wpercent = (basewidth/float(self.img_t5.size[0]))
        hsize = int((float(self.img_t5.size[1])*float(wpercent)))
        self.img_t5 = self.img_t5.resize((basewidth,hsize), Image.ANTIALIAS)
        self.img_t5 = ImageTk.PhotoImage(self.img_t5)
        
        self.img_b1 = Image.open("b1.png")
        wpercent = (basewidth/float(self.img_b1.size[0]))
        hsize = int((float(self.img_b1.size[1])*float(wpercent)))
        self.img_b1 = self.img_b1.resize((basewidth,hsize), Image.ANTIALIAS)
        self.img_b1 = ImageTk.PhotoImage(self.img_b1)
        
        self.img_b2 = Image.open("b2.png")
        wpercent = (basewidth/float(self.img_b2.size[0]))
        hsize = int((float(self.img_b2.size[1])*float(wpercent)))
        self.img_b2 = self.img_b2.resize((basewidth,hsize), Image.ANTIALIAS)
        self.img_b2 = ImageTk.PhotoImage(self.img_b2)
        
        self.img_b3 = Image.open("b3.png")
        wpercent = (basewidth/float(self.img_b3.size[0]))
        hsize = int((float(self.img_b3.size[1])*float(wpercent)))
        self.img_b3 = self.img_b3.resize((basewidth,hsize), Image.ANTIALIAS)
        self.img_b3 = ImageTk.PhotoImage(self.img_b3)
        
        self.img_b4 = Image.open("b4.png")
        wpercent = (basewidth/float(self.img_b4.size[0]))
        hsize = int((float(self.img_b4.size[1])*float(wpercent)))
        self.img_b4 = self.img_b4.resize((basewidth,hsize), Image.ANTIALIAS)
        self.img_b4 = ImageTk.PhotoImage(self.img_b4)
        
        self.img_b5 = Image.open("b5.png")
        wpercent = (basewidth/float(self.img_b5.size[0]))
        hsize = int((float(self.img_b5.size[1])*float(wpercent)))
        self.img_b5 = self.img_b5.resize((basewidth,hsize), Image.ANTIALIAS)
        self.img_b5 = ImageTk.PhotoImage(self.img_b5)
      
    def resetCanvas(self):
        self.C.configure(bg = "grey")
        self.frameWave.update_idletasks()
        time.sleep(0.1)
        self.C.configure(bg = "white")
        self.frameWave.update_idletasks()
        self.refresh()
    
    def refresh(self):
        try:
             while True:
                self.data = Data()
                number = np.arange(50,651,550/14)
                points = np.arange(0,901,900/self.points)
                self.C.delete("all")
                self.Ctop.delete("all")
                self.Cbottom.delete("all")
                self.Ctop.create_text(130,20,fill="white",font="Courier", text="Action performed by user")
                self.Ctop.update_idletasks()
                self.Cbottom.create_text(100,20,fill="white",font="Courier", text="Classifier Output")
                self.Cbottom.update_idletasks()
    
                self.C.create_text(550,number[0],anchor = 'w',text = 'CH1')
                self.C.create_text(550,number[1],anchor = 'w',text = 'CH2')
                self.C.create_text(550,number[2],anchor = 'w',text = 'CH3')
                self.C.create_text(550,number[3],anchor = 'w',text = 'CH4')
                self.C.create_text(550,number[4],anchor = 'w',text = 'CH5')
                self.C.create_text(550,number[5],anchor = 'w',text = 'CH6')
                self.C.create_text(550,number[6],anchor = 'w',text = 'CH7')
                self.C.create_text(550,number[7],anchor = 'w',text = 'CH8')
                self.C.create_text(550,number[8],anchor = 'w',text = 'CH9')
                self.C.create_text(550,number[9],anchor = 'w',text = 'CH10')
                self.C.create_text(550,number[10],anchor = 'w',text = 'CH11')
                self.C.create_text(550,number[11],anchor = 'w',text = 'CH12')
                self.C.create_text(550,number[12],anchor = 'w',text = 'CH13')
                self.C.create_text(550,number[13],anchor = 'w',text = 'CH14')
                self.C.update_idletasks()
                for i in range(self.data.length-1):
                    self.C.create_line(points[i],((self.data.data[0][i]-4100)/200)*self.height + number[0],points[i+1], ((self.data.data[0][i+1]-4100)/200)*self.height + number[0],fill="grey")
                    self.C.create_line(points[i],((self.data.data[1][i]-4100)/200)*self.height + number[1],points[i+1], ((self.data.data[1][i+1]-4100)/200)*self.height + number[1],fill="purple")
                    self.C.create_line(points[i],((self.data.data[2][i]-4100)/200)*self.height + number[2],points[i+1], ((self.data.data[2][i+1]-4100)/200)*self.height + number[2],fill="blue")
                    self.C.create_line(points[i],((self.data.data[3][i]-4100)/200)*self.height + number[3],points[i+1], ((self.data.data[3][i+1]-4100)/200)*self.height + number[3],fill="sea green")
                    self.C.create_line(points[i],((self.data.data[4][i]-4100)/200)*self.height + number[4],points[i+1], ((self.data.data[4][i+1]-4100)/200)*self.height + number[4],fill="gold")
                    self.C.create_line(points[i],((self.data.data[5][i]-4100)/200)*self.height + number[5],points[i+1], ((self.data.data[5][i+1]-4100)/200)*self.height + number[5],fill="orange red")
                    self.C.create_line(points[i],((self.data.data[6][i]-4100)/200)*self.height + number[6],points[i+1], ((self.data.data[6][i+1]-4100)/200)*self.height + number[6],fill="red")
                    self.C.create_line(points[i],((self.data.data[7][i]-4100)/200)*self.height + number[7],points[i+1], ((self.data.data[7][i+1]-4100)/200)*self.height + number[7],fill="brown")
                    self.C.create_line(points[i],((self.data.data[8][i]-4100)/200)*self.height + number[8],points[i+1], ((self.data.data[8][i+1]-4100)/200)*self.height + number[8],fill="chocolate1")
                    self.C.create_line(points[i],((self.data.data[9][i]-4100)/200)*self.height + number[9],points[i+1], ((self.data.data[9][i+1]-4100)/200)*self.height + number[9],fill="aquamarine2")
                    self.C.create_line(points[i],((self.data.data[10][i]-4100)/200)*self.height + number[10],points[i+1], ((self.data.data[10][i+1]-4100)/200)*self.height + number[10],fill="yellow green")
                    self.C.create_line(points[i],((self.data.data[11][i]-4100)/200)*self.height + number[11],points[i+1], ((self.data.data[11][i+1]-4100)/200)*self.height + number[11],fill="SlateGray1")
                    self.C.create_line(points[i],((self.data.data[12][i]-4100)/200)*self.height + number[12],points[i+1], ((self.data.data[12][i+1]-4100)/200)*self.height + number[12],fill="tomato")
                    self.C.create_line(points[i],((self.data.data[13][i]-4100)/200)*self.height + number[13],points[i+1], ((self.data.data[13][i+1]-4100)/200)*self.height + number[13],fill="black")
                    self.C.update_idletasks()
                    print(i)
                
                # serial.write('a')
                # self.Top.after(2000)
                # serial.write('a')
                # self.Top.after(2000)
                # serial.write('a')
                # self.Top.after(2000)


                a = self.data.classifier
                if a == 0 :
                    self.top_img(self.img_t1)
                    self.bottom_img(self.img_b1)
                elif a == 1:
                    self.top_img(self.img_t2)
                    self.bottom_img(self.img_b2)
                elif a == 2:
                    self.top_img(self.img_t3)
                    self.bottom_img(self.img_b3)
                elif a == 3:
                    self.top_img(self.img_t4)
                    self.bottom_img(self.img_b4)
                elif a==4:
                    self.top_img(self.img_t5)
                    self.bottom_img(self.img_b5)
                self.C.update_idletasks()
                self.Top.after(500)    
        except Exception as e:
            print(e)
    def top_img(self,Img1):
        self.Ctop.create_image(300,175,image=Img1,anchor="center");
        self.Ctop.update_idletasks()
    def bottom_img(self,Img2):
        self.Cbottom.create_image(300,175,image=Img2,anchor="center");
        self.Cbottom.update_idletasks()

    
                      
Top = tk.Tk()
window = GUI(Top)
Top.mainloop()