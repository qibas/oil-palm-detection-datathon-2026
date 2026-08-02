import warnings, os, random
warnings.filterwarnings("ignore")
import numpy as np, cv2
from PIL import Image
from grids import load_coco
random.seed(3)
BASE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(BASE,"out")

cats, anns = load_coco("B")
crowns=[a for a in anns if a["cat"]=="Healthy" and (a["bbox"][2]**2+a["bbox"][3]**2)**0.5>120]
sample=random.sample(crowns,9)
cell=200; cols=3; rows=3
canvas=Image.new("RGB",(cols*cell*2,rows*cell),(15,15,15))
for i,a in enumerate(sample):
    im=np.array(Image.open(a["path"]).convert("RGB")).astype(np.float32)
    x,y,w,h=[int(v) for v in a["bbox"]]
    crop=im[max(0,y):y+h, max(0,x):x+w]
    R,G,Bc=crop[...,0],crop[...,1],crop[...,2]
    exg=2*G-R-Bc
    exg8=cv2.normalize(exg,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    _,mask=cv2.threshold(exg8,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    frac=100*mask.mean()/255
    orig=Image.fromarray(crop.astype(np.uint8)).resize((cell,cell),Image.LANCZOS)
    m=Image.fromarray(mask).convert("RGB").resize((cell,cell),Image.NEAREST)
    r,c=divmod(i,cols)
    canvas.paste(orig,(c*cell*2, r*cell))
    canvas.paste(m,(c*cell*2+cell, r*cell))
p=os.path.join(OUT,"SEG_B_crownarea.jpg"); canvas.save(p)
print("saved",p,"(left=crop, right=ExG+Otsu crown mask)")
