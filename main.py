import tkinter as tk
import pandas as pd
from tkinter import PhotoImage, filedialog
from utilities import *
from utilities_transformers import select_model_transformers
tf.autograph.set_verbosity(0)
import tensorflow as tf
from PIL import Image, ImageTk
import time
import threading
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
files_list = []

#Μέθοδος εναλλαγής πλαισίου eg. start frame --> main_frame
def switch_frame(origin,goal):
    goal.pack(fill="both", expand=True)
    if goal!=start_frame:
        root.geometry('1500x850')
        origin.pack_forget()
    else:
        root.geometry('1000x800')
        origin.pack_forget()



#Μέθοδος διάσπασης threads στη CPU  TODO: Optimization for GPU
def thread(machine_learning=True):
    global start_time, t1 
    if machine_learning:
        button.config(state=tk.DISABLED)
        label3.grid(row=7, column=3)
        t2=threading.Thread(target= lambda: print_time(label3, main_frame))
        t2.start()
        t1=threading.Thread(target=run_ML_models)
        t1.start()   
    else:
        start_time = time.time()
        button3.config(state=tk.DISABLED)
        label4.grid(row=7, column=3)
        t2=threading.Thread(target= lambda: print_time(label4, transformers_frame))
        t2.start()
        t1=threading.Thread(target=run_transformers)
        t1.start() 
#Μέθοδος που εμφανίζει και εξαφανίζει το κείμενο Please wait... κάθε ένα δευτερόλεπτο 
def print_time(loading_label, frame):
    if loading_label.cget("text")=="Please wait...":
        loading_label.configure(text="")
    else:
        loading_label.configure(text="Please wait...")
    #αν η thread 1 δεν τρέχει τότε επιστρέφει
    if not t1.is_alive():
        return
    # Εμφανίζει το κείμενο κάθε 1 δευτερόλεπτο καλλώντας τον ευατό της
    frame.after(1000, lambda: print_time(loading_label, frame))

#Μέθοδος για εκτέλεση Machine Learning πειραμάτων
def run_ML_models():
    main_frame.after(0, lambda: print_time(label3, main_frame)) #Ξεκινά την μέθοδο print_time, δηλαδή την t2 thread
    #Συλλογη μεταβλητών
    ans_test_size = ""
    ans_dataset = dataset_var.get()
    ans_class = check_var.get()
    ans_stopword_lemma = stopwords_var.get()
    ans_embeddings = vectorizer_var.get()
    ans_models = model_var.get()
    ans_split = random_split_var.get()
    if ans_models != 'Select a model' and ans_class != "Select Classification" and ans_dataset !='Select Dataset' and ans_stopword_lemma != 'Stopwords/Lemmatization' and ans_embeddings !='Select Vectorizer' and ans_split !='Random Split':
        #Φόρτωση της βάσης δεδομένων
        if ans_dataset == 'politics':
            headers=['Text','Sentiment']
            headers2=['Created_at','Username','Country','City','State_District','Topic','Text','Sentiment']
            politics = pd.read_csv('Politics/balanced_politics.csv', sep=',', names=headers2)
            train_politics = pd.read_csv('Politics/train2682_test810/train_set_politics2682.csv', sep=',', names=headers)
            test_politics = pd.read_csv('Politics/train2682_test810/test_set_politics810.csv', sep=',', names=headers)
        elif ans_dataset == 'skroutz':
            headers=['Text','Sentiment']
            headers2=['id','Text','Sentiment']
            skroutz = pd.read_csv('Skroutz/skroutz_dataset.csv', sep=',', names=headers)
            train_skroutz = pd.read_csv('Skroutz/train_set_skroutz3210.csv', sep=',', names=headers)
            test_skroutz = pd.read_csv('Skroutz/test_set_skroutz1966.csv', sep=',', names=headers)
        else:
            print('Δώστε ένα απο τα παραπάνω dataset!')
        #Προεπεξεργασία
        if ans_dataset == 'politics': 
            if ans_class == 'binary':
                politics = delete_neutral(politics)
                train_politics = delete_neutral(train_politics)
                test_politics = delete_neutral(test_politics)
        if ans_stopword_lemma == 'yes' and ans_dataset == 'politics':
            politics = stopwords_lemmatize(politics)
            train_politics = stopwords_lemmatize(train_politics)
            test_politics = stopwords_lemmatize(test_politics)
        if ans_stopword_lemma == 'yes' and ans_dataset == 'skroutz':
            skroutz=stopwords_lemmatize(skroutz)
            train_skroutz = stopwords_lemmatize(train_skroutz)    
            test_skroutz = stopwords_lemmatize(test_skroutz)
        #Δημιουργία test και train set, αναλόγως την περίπτωση χρήσης
        if ans_dataset == 'politics':
            X = politics['Text']
            y = politics['Sentiment']
            X_train = train_politics['Text']
            y_train = train_politics['Sentiment']
            X_test = test_politics['Text']
            y_test = test_politics['Sentiment']
        else:
            X = skroutz['Text']
            y = skroutz['Sentiment']
            X_train = train_skroutz['Text']
            y_train = train_skroutz['Sentiment']
            X_test = test_skroutz['Text']
            y_test = test_skroutz['Sentiment']
        if ans_embeddings == 'tfidf':
            if ans_split == 'no':
                X_train, X_test, y_train, y_test = tfidf_labelEncoder((X_train, X_test), (y_train, y_test), False)
            else:
                ans_test_size = open_popup(root)
                if ans_test_size != 'error':
                    X_train, X_test, y_train, y_test = tfidf_labelEncoder(X, y, True, ans_test_size) 
                else:
                    label.delete(1.0, "end-1c")
                    label.insert(1.0, str('Please insert 0.2 or 0.3'))
                    button.config(state=tk.NORMAL) 
                    label3.grid_forget()
                    return
        if ans_embeddings == 'word2vec':
            if ans_split == 'no':
                X_train, X_test, y_train, y_test = word2vec_labelEncoder((X_train, X_test), (y_train, y_test), False, ans_dataset)
            else:
                ans_test_size = open_popup(root)
                if ans_test_size != 'error':
                    X_train, X_test, y_train, y_test = word2vec_labelEncoder(X, y, True, ans_dataset, ans_test_size)
                else:
                    clear()
                    label.insert(1.0, str('Please insert 0.2 or 0.3'))
                    button.config(state=tk.NORMAL) 
                    label3.grid_forget()
                    return
        #Εκτέλεση πειράματος
        report, matrix  = select_model(X_train, X_test, y_train, y_test, ans_class, ans_models, ans_embeddings)
        label.delete(1.0, "end-1c")
        label.insert(1.0,f'Model: {ans_models},Dataset: {ans_dataset}, Class: {ans_class}, Stopwords/Lemmatize: {ans_stopword_lemma}, Vectorizer: {ans_embeddings}\n\n\n'+str(report)+'\n'+'Classification Report \n'+matrix) 
        #Αποθήκευση confusion matrix σαν plot.png
        new_image = Image.open('plot.png')
        size = (400, 400)
        scaled_img = new_image.resize(size)
        tk_image2 = ImageTk.PhotoImage(scaled_img)
        label3.grid_forget()
        button.config(state=tk.NORMAL) 
        #αλλαγή της εικόνας στην plot.png
        confusion_matrix_image.configure(image=tk_image2)
        confusion_matrix_image.image = tk_image2
    else:
        label.delete(1.0, "end-1c")
        label.insert(1.0, str('Please select a model and try again!'))
    end_time = time.time()
    runtime = end_time - start_time
    label.insert(1.0,f"Runtime: {round(runtime,2)} seconds \n\n")
    content = label.get("1.0", "end-1c")
    #Αποθήκευση περιοχομένων του text lebel σε txt αρχείο με κατάλληλο όνομα 
    with open("files/"+ans_models+"_"+ans_dataset+"_"+ans_class+"_"+ans_stopword_lemma+"SL_"+ans_embeddings+"_split"+ans_split+str(ans_test_size)+".txt", "w") as file:
        file.write(content)


#Μέθοδος για εκτέλεση Deep Learning πειραμάτων
def run_transformers():
    transformers_frame.after(0, lambda: print_time(label4, transformers_frame))
    #Συλλογη μεταβλητών
    ans_dataset = dataset_var2.get()
    ans_class = class_var.get()
    ans_stopword_lemma = stopwords_var2.get()
    ans_models = transformers_var.get()
    if ans_models != 'Select a model' and ans_class != "Select Classification" and ans_dataset !='Select Dataset' and ans_stopword_lemma != 'Stopwords/Lemmatization':
        #Εκτέλεση πειραμάτων
        report, matrix = select_model_transformers(ans_dataset, ans_class, ans_stopword_lemma, ans_models)
        transformers_label.delete(1.0, "end-1c")
        transformers_label.insert(1.0,f'Model: {ans_models},Dataset: {ans_dataset}, Class: {ans_class}, Stopwords/Lemmatize: {ans_stopword_lemma}\n\n\n'+str(report)+'\n'+'Classification Report \n'+matrix) 
        new_image = Image.open('plot.png')
        size = (400, 400)
        scaled_img = new_image.resize(size)
        tk_image2 = ImageTk.PhotoImage(scaled_img)
        label4.grid_forget()
        button3.config(state=tk.NORMAL) 
        confusion_matrix_image2.configure(image=tk_image2)
        confusion_matrix_image2.image = tk_image2
    else:
        transformers_label.delete(1.0, "end-1c")
        transformers_label.insert(1.0, str('Please select a model and try again!'))
    end_time = time.time()
    runtime = end_time - start_time
    transformers_label.insert(1.0,f"Runtime: {round(runtime,2)} seconds \n\n")
    #TODO:Αποθήκευση περιοχομένων του text lebel σε txt αρχείο με κατάλληλο όνομα 

#Μέθοδος καθαρισμού text label 
def clear(text_label, matrix_label):
    text_label.delete(1.0, "end-1c")
    matrix_label.configure(image=tk_img)
    matrix_label.image = tk_img

#TODO: Μέθοδος φόρτωσης μιας τυχαίας βάσης δεδομένων
def load_csv():
    file_path = filedialog.askopenfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    if file_path:
        txt = str(file_path)
        print(txt)
        df = pd.read_csv(file_path)
        dataset.append(txt)
        print(dataset_box['menu'])
        #label2.insert(1.0, txt)
        #label2.config(state="normal")
        #label2.delete(1.0, "end-1c")
        #label2.insert(1.0, txt)
        #label2.config(state="disabled")
        return df

#Ενα αναδιώμενο παράθυρο, για εισαγωγή του ποσοστού που θα έχει το test set
def open_popup(parent):
    top, entry_var = popUpText(parent)
    parent.wait_window(top)  
    try: 
        value = float(entry_var.get())
    except:
        value = "error"
    return value
#Μέθοδος αποθήκευσης τιμής για το ποσοστό του test set
def popUpText(parent):
    top = tk.Toplevel(parent)
    top.resizable(0,0)
    #τοποθέτηση του αναδιώμενου παραθύρου στο κέντρο της οθόνης
    width = 300
    height = 80
    #μήκος και πλάτος της οθόνης
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    #υπολογισμός x και y συντεταγμένων
    x = (screen_width/2) - (width/4)
    y = (screen_height/2) - (height/2)
    top.geometry('%dx%d+%d+%d' % (width, height, x, y))
    text_label = tk.Label(top, text='Insert a decimal 0.2 or 0.3 to apply the random split')
    text_label.pack()
    entry_var = tk.StringVar()
    entry = tk.Entry(top, textvariable=entry_var)
    entry.pack()
    save_button = tk.Button(top, text="Save", command=lambda: save_entry(entry_var, top))
    save_button.pack()
    return top, entry_var

def save_entry(entry_var, top):
    entry_text = entry_var.get()
    top.destroy()
    return entry_text

    
#------------------------------------Αρχή Εφαρμογής Tkinter------------------------------------------------------

root = tk.Tk()
root.title("K.A.S.S.")
root.geometry("1000x800") 
root.configure(bg='#ADD8E6')
root.resizable(0,0) #ρυθμιση ώστε να μην είναι μεγαλώνει ο χρήστης τις διαστάσεις του παραθύρου

#TODO:Μενού αρχείων και διασύνδεση με τη load_csv μέθοδο 
# Create the menu widget
menu = tk.Menu(root)
root.config(menu=menu)

#Δημιουργία μενού αρχείων
file_menu = tk.Menu(menu)
menu.add_cascade(label="File", menu=file_menu)

#επιλογές μενού αρχείων
file_menu.add_command(label="Open", command=load_csv)
file_menu.add_command(label="Save", command=lambda: print("Save"))
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

#------------------------------------Αρχή βασικού πλαισίου------------------------------------------------------
start_frame = tk.Frame(root)
start_frame.pack(fill="both", expand=True)
title = tk.Label(start_frame, text="International Hellenic University\nSentiment Analyzer", font=("Helvetica", 24), fg='black')
image1 = PhotoImage(file="files/univercity_logo.png")
image_label = tk.Label(start_frame, image=image1)
#label2 = tk.Text(start_frame, wrap="word", state="disabled")
start_button = tk.Button(start_frame,height=2,width=20,activebackground='red', text="Machine Learning", command=lambda: switch_frame(start_frame, main_frame))
load_button = tk.Button(start_frame,height=2,width=20,activebackground='red', text="Deep Learning", command=lambda: switch_frame(start_frame, transformers_frame))
title.grid(row=0, column=1, pady=100, padx=10)
image_label.grid(row=0, column=0, pady=100, padx=10)
#label2.grid(row=1)
load_button.grid(row=4,column=1)
start_button.grid(row=5,column=1)
#------------------------------------Τέλος βασικού πλαισίου------------------------------------------------------


#--------------------------------------------------------Αρχή Machine learning πλαισίου----------------------------------------------------------------

main_frame = tk.Frame(root)

#ορισμός από λίστες για τις υπερπαραμέτρους των πειραμάτων
models = ['rf', 'dt', 'kn', 'mnb', 'lr', 'svm', 'gnb']
stopwords = ['yes', 'no']
binary = ['binary', 'multi']
vectorizer = ['tfidf', 'word2vec']
dataset = ['politics', 'skroutz']

#ορισμός μεταβλητων tkinter τυπου string
model_var = tk.StringVar()
stopwords_var = tk.StringVar()
check_var = tk.StringVar()
dataset_var = tk.StringVar()
vectorizer_var = tk.StringVar()
random_split_var = tk.StringVar()
#Ταμπέλα Please wait...
label3 = tk.Label(main_frame, text='Please wait...')

#ορισμός ταμπέλας πίνακα σύγχησης 
img = Image.open('files/blank.png')
new_size = (400, 400)
scaled_image = img.resize(new_size)
tk_img = ImageTk.PhotoImage(scaled_image)
confusion_matrix_image = tk.Label(main_frame,image=tk_img)

#Προεπιλεγμένες τιμες
model_var.set("Select Model")
stopwords_var.set('Stopwords/Lemmatization')
check_var.set("Select Classification")
dataset_var.set("Select Dataset")
vectorizer_var.set("Select Vectorizer")
random_split_var.set("Random Split")

#Περιγραφικές ταμπέλες
dropdown_label = tk.Label(main_frame, text="Choose a Model:", font=("Helvetica", 12), bg='#ADD8E6')
stopwords_box_label = tk.Label(main_frame, text="Select [yes] to remove Stopwords and apply Lemmatization:", font=("Helvetica", 12), bg='#ADD8E6')
check_box_label = tk.Label(main_frame, text="Select Binary or Trinary Classification:", font=("Helvetica", 12), bg='#ADD8E6')
dataset_box_label = tk.Label(main_frame, text="Select dataset: ", font=("Helvetica", 12), bg='#ADD8E6')
vectorizer_box_label = tk.Label(main_frame, text="Select vectorizer: ", font=("Helvetica", 12), bg='#ADD8E6')
random_split_label = tk.Label(main_frame, text="Select [yes] to apply random split: ", font=("Helvetica", 12), bg='#ADD8E6')

#Ταμπέλες πολλαπλών επιλογών
check_box = tk.OptionMenu(main_frame, check_var, *binary)
dataset_box = tk.OptionMenu(main_frame, dataset_var, *dataset)
model_box = tk.OptionMenu(main_frame, model_var, *models)
stopwords_box = tk.OptionMenu(main_frame, stopwords_var,*stopwords )
vectorizer_box = tk.OptionMenu(main_frame,vectorizer_var,*vectorizer)
random_split_box = tk.OptionMenu(main_frame, random_split_var, *stopwords)

#Ταμπέλα text
label = tk.Text(main_frame, height=20, width=80, padx=2,pady=15)

#Κουμπια έναρξης, επιστροφής και εκαθάρισης 
clear_button = tk.Button(main_frame, text="Delete All", command=lambda: clear(label, confusion_matrix_image), bg='red', fg='white', font=("Helvetica", 12))
button = tk.Button(main_frame, text="Print", command=thread, bg='green', fg='white', font=("Helvetica", 12))
button2 = tk.Button(main_frame, text="Back", command=lambda: switch_frame(main_frame, start_frame), bg='blue', fg='white', font=("Helvetica", 12))

 

#Παράθεση όλων των ταμπέλων στην οθόνη με χρήση του συστήματος πλέγματος(grid) του tkinter
dropdown_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
model_box.grid(row=0, column=1, padx=10, pady=10)
check_box_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
check_box.grid(row=1, column=1, padx=10, pady=10)
vectorizer_box_label.grid(row=2, column=0, padx=10, pady=10,sticky="w")
vectorizer_box.grid(row=2, column=1, padx=10, pady=10)
dataset_box_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")
dataset_box.grid(row=3, column=1, padx=10, pady=10)
stopwords_box_label.grid(row=4, column=0, padx=10, pady=10, sticky="w")
stopwords_box.grid(row=4, column=1, padx=10, pady=10)
random_split_label.grid(row=5, column=0, padx=10, pady=10, sticky="w")
random_split_box.grid(row=5, column=1, padx=10, pady=10)
label.grid(row=6, column=0, columnspan=2,padx=10, pady=10)
confusion_matrix_image.grid(row=6, column=3)
clear_button.grid(row=7, column=0,padx=10, pady=10, sticky="w")
button.grid(row=7, column=2)
button2.grid(row=8, column=0,padx=10, pady=10, sticky="w")
#--------------------------------------------------------Τέλος Machine learning πλαισίου----------------------------------------------------------------


#--------------------------------------------------------Αρχή Deep learning /  transformers πλαισίου----------------------------------------------------------------
transformers_frame = tk.Frame(root)

#ορισμός από λίστες για τις υπερπαραμέτρους των πειραμάτων
transformers = ['bert', 'distilbert', 'roberta', 'gpt']
transformers_var = tk.StringVar()
stopwords_var2 = tk.StringVar()
class_var = tk.StringVar()
dataset_var2 = tk.StringVar()

#Ταμπέλα Please wait...
label4 = tk.Label(transformers_frame, text='Please wait...')

#ορισμός ταμπέλας πίνακα σύγχησης 
confusion_matrix_image2 = tk.Label(transformers_frame,image=tk_img)

#Προεπιλεγμένες τιμες
transformers_var.set("Select Model")
stopwords_var2.set('Stopwords/Lemmatization')
class_var.set("Select Classification")
dataset_var2.set("Select Dataset")

#Περιγραφικές ταμπέλες
dropdown_label2 = tk.Label(transformers_frame, text="Choose a Model:", font=("Helvetica", 12), bg='#ADD8E6')
stopwords_box_label2 = tk.Label(transformers_frame, text="Select [yes] to remove Stopwords and apply Lemmatization:", font=("Helvetica", 12), bg='#ADD8E6')
check_box_label2 = tk.Label(transformers_frame, text="Select Binary or Trinary Classification:", font=("Helvetica", 12), bg='#ADD8E6')
dataset_box_label2 = tk.Label(transformers_frame, text="Select dataset: ", font=("Helvetica", 12), bg='#ADD8E6')

#Ταμπέλες πολλαπλών επιλογών
class_box = tk.OptionMenu(transformers_frame, class_var, *binary)
dataset_box2 = tk.OptionMenu(transformers_frame, dataset_var2, *dataset)
model_box2 = tk.OptionMenu(transformers_frame, transformers_var, *transformers)
stopwords_box2 = tk.OptionMenu(transformers_frame, stopwords_var2,*stopwords )

#Ταμπέλα text
transformers_label = tk.Text(transformers_frame, height=20, width=80, padx=2,pady=15)

#Κουμπια έναρξης, επιστροφής και εκαθάρισης 
clear_button2 = tk.Button(transformers_frame, text="Delete All", command=lambda: clear(transformers_label, confusion_matrix_image2), bg='red', fg='white', font=("Helvetica", 12))
button3 = tk.Button(transformers_frame, text="Print", command=lambda: thread(machine_learning=False), bg='green', fg='white', font=("Helvetica", 12))
button4 = tk.Button(transformers_frame, text="Back", command=lambda: switch_frame(transformers_frame, start_frame), bg='blue', fg='white', font=("Helvetica", 12))

#Παράθεση όλων των ταμπέλων στην οθόνη με χρήση του συστήματος πλέγματος(grid) του tkinter
dropdown_label2.grid(row=0, column=0, padx=10, pady=10, sticky="w")
model_box2.grid(row=0, column=1, padx=10, pady=10)
check_box_label2.grid(row=1, column=0, padx=10, pady=10, sticky="w")
class_box.grid(row=1, column=1, padx=10, pady=10)
dataset_box_label2.grid(row=2, column=0, padx=10, pady=10, sticky="w")
dataset_box2.grid(row=2, column=1, padx=10, pady=10)
stopwords_box_label2.grid(row=3, column=0, padx=10, pady=10, sticky="w")
stopwords_box2.grid(row=3, column=1, padx=10, pady=10)
transformers_label.grid(row=4, column=0, columnspan=2,padx=10, pady=10)
confusion_matrix_image2.grid(row=4, column=3)
clear_button2.grid(row=5, column=0,padx=10, pady=10, sticky="w")
button3.grid(row=5, column=2)
button4.grid(row=6, column=0,padx=10, pady=10, sticky="w")
#--------------------------------------------------------Τέλος Deep learning /  transformers πλαισίου----------------------------------------------------------------


root.mainloop()
