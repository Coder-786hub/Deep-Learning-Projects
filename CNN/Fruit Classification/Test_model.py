from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import joblib
import cv2


model = joblib.load("fruit_model.joblib")


# Preprocessing Image
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

def getClassNo(classNo):
    if classNo == 0 : return "Apple Braeburn"
    elif classNo == 1 : return  "Apple Granny Smith"
    elif classNo == 2 : return  "Apricot"
    elif classNo == 3 : return  "Avocado"
    elif classNo == 4 : return  "Banana"
    elif classNo == 5 : return  "Blueberry"
    elif classNo == 6 : return  "Cactus fruit"
    elif classNo == 7 : return  "Cantaloupe"
    elif classNo == 8 : return  "Cherry"
    elif classNo == 9 : return  "Clementine"
    elif classNo == 10 : return  "Corn"
    elif classNo == 11 : return  "Cucumber Ripe"
    elif classNo == 12 : return  "Grape Blue"
    elif classNo == 13 : return  "Kiwi"
    elif classNo == 14 : return  "Lemon"
    elif classNo == 15 : return  "Limes"
    elif classNo == 16 : return  "Mango"
    elif classNo == 17 : return  "Onion White"
    elif classNo == 18 : return  "Orange"
    elif classNo == 19 : return  "Papaya"
    elif classNo == 20 : return  "Passion Fruit"
    elif classNo == 21 : return  "Peach"
    elif classNo == 22 : return  "Pear"
    elif classNo == 23 : return  "Pepper Green"
    elif classNo == 24 : return  "Pepper Red"
    elif classNo == 25 : return  "Pineapple"
    elif classNo == 26 : return  "Plum"
    elif classNo == 27 : return  "Pomegranate"
    elif classNo == 28 : return  "Potato Red"
    elif classNo == 29 : return  "Raspberry"
    elif classNo == 30 : return  "Strawberry"
    elif classNo == 31 : return  "Tomato"
    elif classNo == 32 : return  "Watermelon"
        

def check():
    if filepath:
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(100, 100))
        img = preprocessing(img)
        img = img.reshape((1, 100, 100, 1))

        # Predict the using loaded image
        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis = 1)[0]
        hightProvibility = np.max(predictions)

        result_label.config(text = f"Detected: {getClassNo(classIndex)} with confidence{hightProvibility:.2f}")
        upload_btn.config(text = "UPLOAD", command = showimage)
        
        
def showimage():
    global filepath

    file = filedialog.askopenfilename()
    if file:
        filepath = file
        img = Image.open(file)
        photo = img.resize((295, 295))
        photo1 = ImageTk.PhotoImage(photo)
        output_label.config(image = photo1)
        output_label.image = photo1
        upload_btn.config(text = "CHECK", command = check)



app = Tk()
app.geometry("800x700+100+50")
app.title("Fruit And Vegetable Classifier")
app.config(bg = "lightgreen")
app.resizable(0,0)

heading = Label(app, text = "FRUIT AND VEGETABLE CLASSIFIER", font = ("ROBOT", 30, "bold", "underline"), fg = "blue", bg = "lightgreen")
heading.pack(fill= "x", pady = 10)


frame = Frame(app, width = 300, height = 300)
frame.pack()

output_label = Label(frame, bg = "black", width = 295, height = 295, pady = 50)
output_label.place(x = 0, y = 0)


upload_btn = Button(app, text = "UPLOAD", font = ("ROBOT", 25, "bold"), fg = "red", bg = "green", bd = 8, relief = "groove", command = showimage)
upload_btn.pack(pady = 50)

result_label = Label(app, fg = "red", bg = "lightgreen", font = ("ROBOT", 20, "bold"))
result_label.pack(pady = 30)


app.mainloop()