import cv2, numpy as np
im=cv2.imread('foto/parete_originale.jpg'); H,W=im.shape[:2]
lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB).astype(np.float32)
L,a,b=cv2.split(lab)
NICCHIE=[
 [(202,97),(513,116),(545,951),(249,951)],
 [(778,142),(1059,157),(1059,625),(788,622)],
 [(1265,180),(1517,197),(1507,844),(1260,844)],
 [(788,706),(1060,708),(1063,1249),(792,1249)],
 [(332,1356),(1444,1316),(1440,1562),(345,1623)],
]
def polymask(polys):
  m=np.zeros((H,W),np.uint8); cv2.fillPoly(m,[np.array(p,np.int32) for p in polys],1); return m
wallpoly=polymask([[(40,0),(1645,0),(1592,2010),(1592,2576),(0,2576),(0,2100),(172,2095)]])
nm=polymask(NICCHIE)
nm_big=cv2.dilate(nm,np.ones((25,25),np.uint8))
ca,cb=a-128,b-128; chroma=np.hypot(ca,cb); ang=np.degrees(np.arctan2(cb,ca))
base=(a>150)&(b>146)
ombra=(nm_big>0)&(L<130)&(chroma>18)&(ang>25)&(ang<52)
vetro=polymask([NICCHIE[0],NICCHIE[2]])
vetro_r=(vetro>0)&(chroma>8)&(ang>18)&(ang<58)
basso=polymask([[(870,2150),(1110,2150),(1110,2576),(870,2576)],[(1345,2100),(1592,2100),(1592,2200),(1415,2576),(1345,2576)]])
basso_r=(basso>0)&(b>140)&(a>149)&(chroma>18)&(ang>30)&(ang<55)
sedia=polymask([[(1100,2150),(1420,2150),(1420,2576),(1100,2576)]])
chair=polymask([[(1095,2040),(1390,2040),(1390,2200),(1350,2330),(1300,2420),(1255,2576),(1110,2576),(1095,2400)]])
sedia_r=(sedia>0)&(chair==0)&(L<95)&(a>148)&(b>140)
m=((base|ombra|vetro_r|basso_r|sedia_r)&(wallpoly>0)).astype(np.uint8)
m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
soft=cv2.GaussianBlur(m.astype(np.float32),(0,0),1.5)

# colore target verde salvia
t=cv2.cvtColor(np.uint8([[[0x88,0xaf,0x9c]]]),cv2.COLOR_BGR2LAB)[0,0].astype(np.float32)
wm=(m>0)&(nm_big==0)
Lm=L[wm].mean(); cm=chroma[wm].mean()
tc=np.hypot(t[1]-128,t[2]-128)
k=np.clip(chroma/cm,0,1.6)
L2=np.clip(L*(t[0]/Lm),0,255)
a2=128+(t[1]-128)*k; b2=128+(t[2]-128)*k
def merge(L2,a2,b2,soft):
  o=np.dstack([L*(1-soft)+L2*soft, a*(1-soft)+a2*soft, b*(1-soft)+b2*soft])
  return cv2.cvtColor(np.clip(o,0,255).astype(np.uint8),cv2.COLOR_LAB2BGR)
verde=merge(L2,a2,b2,soft)
cv2.imwrite('foto/parete_verde_salvia.jpg',verde,[cv2.IMWRITE_JPEG_QUALITY,93])

# versione 2: incavature nere + luci soffuse
inside=cv2.GaussianBlur(nm.astype(np.float32),(0,0),1.2)
solido=cv2.GaussianBlur(((base|ombra)&~(vetro_r&~base)).astype(np.float32),(0,0),1.2)
vt=cv2.GaussianBlur((vetro_r&~base&~ombra).astype(np.float32),(0,0),1.2)
Ld=8+0.10*L; Lg=0.5*L
Lin=Ld*solido+Lg*vt+L2*(1-solido-vt).clip(0,1)
# nicchie con vetro satinato: peso continuo di 'colore parete'
wcont=np.clip((chroma-6)/16,0,1)*np.clip(1-np.abs(ang-38)/30,0,1)
wcont=cv2.GaussianBlur(wcont.astype(np.float32),(0,0),2.0)
wcont=np.maximum(wcont,solido)
Lv=L*(1-wcont)+Ld*wcont
vm=cv2.GaussianBlur(vetro.astype(np.float32),(0,0),1.2)
Lin=Lin*(1-vm)+Lv*vm
Ln=L2*(1-inside)+Lin*inside
an=a2*(1-inside)+128*inside; bn=b2*(1-inside)+129*inside
img=merge(Ln,an,bn,soft).astype(np.float32)/255
lin=img**2.2
warm=np.array([0.55,0.78,1.0],np.float32)  # BGR caldo
glow=np.zeros((H,W),np.float32); led=np.zeros((H,W),np.float32)
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
for p in NICCHIE:
  p=np.array(p,np.float32)
  x0=(p[0,0]+p[3,0])/2; x1=(p[1,0]+p[2,0])/2
  y0=(p[0,1]+p[1,1])/2; y1=(p[2,1]+p[3,1])/2
  w=x1-x0; h=y1-y0; cx=(x0+x1)/2
  mk=polymask([p.astype(np.int32)]).astype(np.float32)
  mk=cv2.GaussianBlur(mk,(0,0),2)
  ytop=y0+min(h*0.06,40)
  d=((xx-cx)/(w*0.62))**2+((yy-ytop)/(h*0.95))**2
  g=np.exp(-d*2.2)*(yy>ytop-30)
  glow+=g*mk*0.55
  # striscia LED
  s=np.exp(-((yy-ytop)/5)**2)*np.clip(1-np.abs(xx-cx)/(w*0.46),0,1)**0.3*mk
  led+=s
  # luce sul piano d'appoggio
  pl=np.exp(-((yy-(y1-8))/(h*0.06))**2)*np.exp(-((xx-cx)/(w*0.5))**2)*mk
  glow+=pl*0.12
  # alone esterno sulla parete
halo=cv2.GaussianBlur(polymask(NICCHIE).astype(np.float32),(0,0),35)*(1-polymask(NICCHIE))
glow_tot=glow[...,None]*warm
lin=lin*(1+glow[...,None]*3.0*warm)+glow_tot*0.22+led[...,None]*warm*1.2+halo[...,None]*warm*0.05
img=np.clip(lin,0,1)**(1/2.2)
cv2.imwrite('foto/parete_salvia_nicchie_nere_luci.jpg',(img*255).astype(np.uint8),[cv2.IMWRITE_JPEG_QUALITY,93])
