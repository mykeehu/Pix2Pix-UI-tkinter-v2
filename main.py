import io
import cv2
import time
import numpy.distutils.cpuinfo
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib
import niqe as ni
import shutil
import imquality.brisque as brisque
import warnings
warnings.filterwarnings("ignore")         # Ignore Kit Warnings
from tkinter import filedialog
#from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity
from PIL import Image
from PIL import ImageTk
from pynvml import *
matplotlib.use("Agg")
import tensorflow as tf

EPOCH_SAVE = 500

def MKDIR(Dir):
	if not os.path.isdir(Dir):
		os.mkdir(Dir)
MKDIR("imgset")
MKDIR("imgset/from")
MKDIR("imgset/target")
MKDIR("imgset/train")

#Limit GPU VRAM usage to only 75% available
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.75)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)

#Introduce the Pix2Pix library
import pix2pix_512 as pix

# Create Tkinter window objects
app = tk.Tk()
entryText_1 = tk.StringVar()
entryText_2 = tk.StringVar()
entryText_3 = tk.StringVar()
#1 execution continued, training function
class Threader(threading.Thread):

    def __init__(self, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.daemon = True
        self.start()

    def run(self):
        path = entryText_1.get()
        epoch = int(entryText_2.get())
        fit(epoch, path)

# Second execution, generate image
class Threader_2(threading.Thread):
    def __init__(self, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.daemon = True
        self.start()

    def run(self):
        generate_save_images(int(entryText_3.get()), entryText_1.get())

# The third execution continues, calculating the IQA value
class Threader_3(threading.Thread):
    def __init__(self, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.daemon = True
        self.start()

    def run(self):
        IQA_cal()

# Acceptable image formats
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

#Verify if it is an image
def is_image_file(file):
    return any(file.endswith(extension) for extension in IMG_EXTENSIONS)

# Create a List with images
def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

#Subfolder Confirmation
def select_folder():
    dirname = filedialog.askdirectory()
    pix.set_path(dirname+"\\")
    if len(dirname) > 0:
        entryText_1.set(dirname + "\\")
        pix.set_path(dirname+"\\")
        ok_str = ".....OK"
        fail_str = ".....Not found"
        if os.path.isdir(dirname + "\\test"):
            test = make_dataset(dirname + "\\test")
            test_path.config(text="Image test" + ok_str + "  " + str(len(test)) + " images")
            gene_image.config(text="Image test \nmax {}".format(len(test)))
            gene_num.config(state=tk.ACTIVE)
        else:
            test_path.config(text="Image test" + fail_str)

        if os.path.isdir(dirname + "\\train"):
            train = make_dataset(dirname + "\\train")
            train_path.config(text="Image train" + ok_str + "  " + str(len(train)) + " images")
            check_epoch.config(state=tk.ACTIVE)
        else:
            train_path.config(text="Image train" + fail_str)

        if os.path.isdir(dirname + "\\save"):
            for _, _, file in os.walk(os.path.join(dirname + "\\save\\")):
                index = str(file[-3:-2])[3:5]
                if len(index) > 0:
                    Model_path.config(text="Model save" + ok_str + "\nLatest Model is : " + index)
                else:
                    Model_path.config(text="Model save" + ok_str + "\nLatest Model is : None")
        else:
            Model_path.config(text="Model save" + fail_str)

        if os.path.isdir(dirname + "\\gt") and os.path.isdir(dirname + "\\gene"):
            gene_save.config(text="gt & gene" + ok_str)
            dataset_1 = make_dataset(dirname + "\\gene\\")
            dataset_2 = make_dataset(dirname + "\\gt\\")

            if len(dataset_1) > 0 and len(dataset_2) > 0:
                psnr_calculation.config(state=tk.ACTIVE)
            else:
                psnr_calculation.config(state=tk.DISABLED)
        else:
            gene_save.config(text="image_save" + fail_str)

# Set the number of training sessions
def set_epochs():
    epoch = entryText_2.get()
    if epoch.isdigit():
        if len(epoch) > 0 and int(epoch) > 0 and len(entryText_1.get()) > 0:
            able_train()
            s = "epochs"
            train_epochs.config(text=(s + " : " + epoch))
    else:
        train_epochs.config(text="Enter number")
        disable_train()
        check_epoch.configure(state=tk.ACTIVE)

#Training Features
def fit(epochs, path):
    total_time = 0.0
    row = 0
    Image_folder.config(state=tk.DISABLED)
    gene_num.config(state=tk.DISABLED)
    Gene_GT .config(state=tk.DISABLED)
    disable_train()
    pix.ckpt_restore(path)
    process_box.delete(0, tk.END)
    train_ds, test_ds = pix.create_dataset(path)
    model_save = path + "save\\"
    state.configure(text="State.....ready")

    process_box_2.config(state='normal')
    process_box_2.delete("1.0", "end")
    for epoch in range(epochs):
        start = time.time()
        for example_input, example_target in test_ds.take(1):
          generate_images(pix.generator, example_input, example_target)
        state.configure(text="State.....Training")
        for n, (input_image, target) in train_ds.enumerate():
            process_box_2.insert('end', '*')
            pix.train_step(input_image, target, epoch)
        process_box_2.delete("1.0", "end")
        cost = round(time.time() - start, 2)
        #if (epoch + 1) % 20 == 0:
        if (epoch + 1) % EPOCH_SAVE == 0:
            pix.checkpoint.save(file_prefix=model_save)
            time_c = 'Time taken for epoch {} is {} sec.......Save model\n'.format(epoch + 1, cost)
        else:
            time_c = 'Time taken for epoch {} is {} sec\n'.format(epoch + 1, cost)
        total_time += cost
        process_box.insert(row, time_c)
        process_box.see(row)
        row += 1
    process_box.insert(row, "\nEpochs finish\n Total cost {} sec".format(round(total_time, 2)))
    process_box.see(row)
    process_box_2.config(state='disabled')
    state.configure(text="State.....Finish")
    time.sleep(2)
    state.configure(text="State.....Waiting")
    Image_folder.config(state=tk.ACTIVE)
    gene_num.config(state=tk.ACTIVE)
    if len(entryText_3.get()) > 0 and entryText_3.get().isdigit() == True:
        Gene_GT .config(state=tk.ACTIVE)
    able_train()

#Generate training previews
def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Target', 'Predicted Image']

    for i in range(3):
      plt.subplot(1, 3, i+1)
      plt.title(title[i])
      plt.imshow(display_list[i] * 0.5 + 0.5)
      plt.axis('off')
    size = 572, 194
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    axc = Image.open(buf)
    im1 = axc.crop((150, 530, 1400, 950))
    im1.thumbnail(size)

    imgtk = ImageTk.PhotoImage(image=im1)
    labelLogo.configure(image=imgtk)
    labelLogo.image = imgtk
    buf.close()

#MasterGenerator function
def generata_save(model, test_input, tar, count, path):
    count_len = len(str(count))
    if count_len == 1:
        name = "000" + str(count)
    elif count_len == 2:
        name = "00" + str(count)
    elif count_len == 3:
        name = "0" + str(count)
    else:
        name = str(count)

    prediction = model(test_input, training=True)
    tf.keras.preprocessing.image.save_img(os.path.join(path + "gene\\" + name + "_gene.png"), prediction[0], file_format='PNG')
    tf.keras.preprocessing.image.save_img(os.path.join(path + "gt\\" + name + "_gt.png"), tar[0], file_format='PNG')
    save_place = 'plot_' + str(count) + '.png'
    process_box.insert(count, save_place)
    process_box.see(count+1)
    psnr_calculation.config(state=tk.ACTIVE)

#Generate test images
def generate_save_images(NUM, path):
    Gene_GT.config(state=tk.DISABLED)
    _, test_ds = pix.create_dataset(path)
    pix.ckpt_restore(path)
    count = 0
    process_box.delete(0, tk.END)
    state.configure(text="State.....ready")

    shutil.rmtree(path + "gene\\")
    shutil.rmtree(path + "gt\\")
    os.mkdir(path + "gene\\")
    os.mkdir(path + "gt\\")

    start = time.time()
    for inp, tar in test_ds.take(NUM):
        state.configure(text="State.....Generating")
        generata_save(pix.generator, inp, tar, count, path)
        count += 1
    print(time.time()-start)
    process_box.insert(count, "\nFinish\n")
    Gene_GT.config(state=tk.ACTIVE)
    state.configure(text="State.....Finish")
    time.sleep(1)
    state.configure(text="State.....Waiting")

# Set the number of times to generate
def set_gene():
    num = entryText_3.get()
    if num.isdigit():
        if int(num) > 0 and len(entryText_3.get()) > 0:
            Gene_GT.config(state=tk.ACTIVE)

# Calculation of IQA values
def IQA_cal():
    process_box.delete(0, tk.END)
    path = entryText_1.get()

    gene_img = path + "gene\\"
    gt_img = path + "gt\\"
    img = make_dataset(gene_img)
    ori = make_dataset(gt_img)
    IQA_num_list = [0.0, 0.0, 0.0, 0.0]  #psnr ssim brisque niqe
    count = 0

    for gene, gt_img in zip(img, ori):
        state.configure(text="State.....Calculating")
        dehaze = cv2.imread(gene)
        gt = cv2.imread(gt_img)

        BRI = brisque.score(dehaze)
        psnr = cv2.PSNR(gt, dehaze)
        #ssim = compare_ssim(gt, dehaze, multichannel=True)
        ssim = structural_similarity(gt, dehaze, multichannel=True)
        niqe_num = ni.niqe(dehaze)

        result = gene.split("\\")[-1]
        result += "...done"
        process_box.insert(count, result)
        process_box.see(count)

        IQA_num_list[0] += psnr
        IQA_num_list[1] += ssim
        if BRI >= 0:
            IQA_num_list[2] += BRI
        IQA_num_list[3] += niqe_num
        count += 1

    str_list = ['PSNR avg : ', 'SSIM avg :', 'BRI avg : ', 'niqe avg : ']
    total = len(img)

    for num, strz in zip(IQA_num_list, str_list):
        process_box.insert(count, strz + str(round(num / total, 4)))
        process_box.see(count)
        count += 1

    process_box.see(count+1)
    state.configure(text="State.....Waiting")

#Close window
def close_window():
    running = threading.Event()
    running.set()
    running.clear()
    app.destroy()

#Disable Training Button
def disable_train():
    Train.configure(state=tk.DISABLED)
    check_epoch.configure(state=tk.DISABLED)

#Enable Training Button
def able_train():
    Train.configure(state=tk.ACTIVE)
    check_epoch.configure(state=tk.ACTIVE)

#Get CPU, GPU information
nvmlInit()
MyCPU = numpy.distutils.cpuinfo.cpu.info[0]
handle = nvmlDeviceGetHandleByIndex(0)

app.title("Pix2Pix-------GPU:" + str(nvmlDeviceGetName(handle))[2:-1] + "---------|---------CPU: " + MyCPU['ProcessorNameString'])   #Display the use of CPU and GPU
app.geometry('925x750')            #Window size 925*750
app.resizable(width=0, height=0)   #Non-variable window size

textbox_1 = tk.Entry(app, text=entryText_1, borderwidth=3, relief="sunken", width=29)    #Top left box
textbox_2 = tk.Entry(app, text=entryText_2, borderwidth=3, relief="sunken", width=29)    #2 top left frame
textbox_3 = tk.Entry(app, text=entryText_3, borderwidth=3, relief="sunken", width=29)    #3 boxes on the top left
time_record_title = tk.Label(app, text="Time record", height=1, width=10, compound="left")
working_title = tk.Label(app, text="Working print", height=1, width=10, compound="left")
state = tk.Label(app, text="State.....Waiting", height=1, width=46, borderwidth=3, relief="sunken", compound="left")
process_box = tk.Listbox(app, borderwidth=3, width=46, height=28, relief="sunken")
process_box_2 = tk.Text(app, state='disabled', borderwidth=3, width=43, height=34, relief="sunken")
psnr_calculation = tk.Button(app, text="IQA", height=1, width=42, compound="c", command=lambda: Threader_3(name='Thread-name'))

time_record_title.place(x=250, y=230)   # "Time Record" words
process_box.place(x=250, y=255)         #Set the position of the white box on the left
working_title.place(x=600, y=230)       #"Working print" characters
process_box_2.place(x=598, y=255)       #Set the position of the white box on the right
state.place(x=250, y=725)               #Status Bar Settings
psnr_calculation.place(x=602, y=720)    #IQA value calculation button position setting

im = Image.open("preview.png")
igtk = ImageTk.PhotoImage(image=im)
labelLogo = tk.Label(app, borderwidth=3, relief="sunken", image=igtk, width=650, height=194)
labelLogo.place(x=250, y=20)

space = tk.Label(app, text="Pix2Pix GUI ver.", height=3, width=30, compound="c")
Image_folder = tk.Button(app, text="Load Main Folder", command=select_folder, height=3, width=15, compound="c")
Image_path = tk.Label(app, text="Images Path", height=3, width=15, compound="c")

test_path = tk.Label(app, text="Image test.....Waiting", height=3, width=25, compound="c")
train_path = tk.Label(app, text="Image train.....Waiting", height=3, width=25, compound="c")
Model_path = tk.Label(app, text="Model save.....Waiting", height=3, width=25, compound="c")
gene_save = tk.Label(app, text="gt & gene.....Waiting", height=3, width=25, compound="c")

check_epoch = tk.Button(app, text="Set_epoch", command=set_epochs, height=2, width=9, compound="c")
train_epochs = tk.Label(app, text="epochs", height=3, width=15, compound="c")
Train = tk.Button(app, text="Start Train", height=3, width=15, compound="c", command=lambda: Threader(name='Thread-name'))

gene_image = tk.Label(app, text="gene Image", height=3, width=18, compound="c")
gene_num = tk.Button(app, text="Set_num", command=set_gene, height=2, width=9, compound="c")
Gene_GT = tk.Button(app, text="Generate Image", height=3, width=15, compound="c", command=lambda: Threader_2(name='Thread-name'))

button = tk.Button(app, text="Quit", command=close_window, width=15, compound="c")

disable_train()
check_epoch.config(state=tk.DISABLED)
gene_num.config(state=tk.DISABLED)
Gene_GT.config(state=tk.DISABLED)
psnr_calculation.config(state=tk.DISABLED)
textbox_1.configure(state='readonly')
process_box_2.config(state='disabled')

space.grid(column=0, row=0, ipadx=1, pady=1, sticky=tk.W)              #Blank
Image_folder.grid(column=0, row=1, ipadx=50, pady=1, sticky=tk.W)      #select folder button
Image_path.grid(column=0, row=2, ipadx=50, pady=2, sticky=tk.W+tk.S)   #Display "Images Path"
textbox_1.grid(column=0, row=3, padx=5, pady=2, sticky=tk.W+tk.S)      #Display the main folder path
test_path.grid(column=0, row=4, padx=10, pady=2, sticky=tk.W+tk.S)     #Test folder confirmation
train_path.grid(column=0, row=5, padx=10, pady=2, sticky=tk.W+tk.S)    #Training folder confirmation
Model_path.grid(column=0, row=6, padx=10, pady=2, sticky=tk.W+tk.S)    #Model folder confirmation
gene_save.grid(column=0, row=7, padx=10, pady=2, sticky=tk.W+tk.S)     #Generate folder confirmation
train_epochs.grid(column=0, row=8, padx=10, pady=1, sticky=tk.W+tk.S)  #training area
check_epoch.grid(column=0, row=8, padx=140, pady=1, sticky=tk.W+tk.S)  #Training count setting
textbox_2.grid(column=0, row=9, padx=5, pady=2, sticky=tk.W+tk.S)      #Enter the number of training sessions
Train.grid(column=0, row=10, ipadx=50, pady=5, sticky=tk.W+tk.S)       #Training start button
gene_image.grid(column=0, row=11, padx=10, pady=1, sticky=tk.W+tk.S)   #Testing area
gene_num.grid(column=0, row=11, padx=140, pady=1, sticky=tk.W+tk.S)    #test generation sheets
textbox_3.grid(column=0, row=12, padx=5, pady=2, sticky=tk.W+tk.S)     #Enter the number of sheets
Gene_GT.grid(column=0, row=13, ipadx=50, pady=5, sticky=tk.W+tk.S)     #set the number of generation
button.grid(column=0, row=14, ipadx=50, pady=8, sticky=tk.W+tk.S)      #Start test image generation
app.mainloop()                                                         #Start GUI
