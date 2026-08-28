import pypdf, re, sys, zlib, struct, numpy as np

PDF="/Users/nizam/Library/Containers/net.whatsapp.WhatsApp/Data/tmp/documents/9A439A15-47B4-40A7-BFBF-C20E87FFF5AC/An_Overview_of_the_JPEG_AI_Learning-Based_Image_Coding_Standard.pdf"

def mul(a,b):  # 2x3 affine  [a b c d e f]
    a0,a1,a2,a3,a4,a5=a; b0,b1,b2,b3,b4,b5=b
    return (a0*b0+a1*b2, a0*b1+a1*b3, a2*b0+a3*b2, a2*b1+a3*b3, a4*b0+a5*b2+b4, a4*b1+a5*b3+b5)
def app(m,x,y):
    return (m[0]*x+m[2]*y+m[4], m[1]*x+m[3]*y+m[5])

NUM=r'-?\d*\.?\d+'
TOK=re.compile(rb'(?:'+NUM.encode()+rb')|[A-Za-z\'"*]+|/[^\s/\[\]<>()]+|\[|\]|<<|>>|\(|\)|<|>')

def parse(data):
    """yield (op, args) skipping text/inline-image content."""
    toks=[]; i=0; n=len(data)
    out=[]
    for m in TOK.finditer(data):
        t=m.group()
        try:
            toks.append(float(t)); continue
        except ValueError: pass
        s=t.decode('latin-1')
        if s and (s[0].isalpha() or s in ("'",'"')):
            out.append((s,[x for x in toks if isinstance(x,float)])); toks=[]
        else:
            toks=[]
    return out

def flatten_cubic(p0,p1,p2,p3,n=8):
    pts=[]
    for k in range(1,n+1):
        t=k/n; mt=1-t
        x=mt**3*p0[0]+3*mt*mt*t*p1[0]+3*mt*t*t*p2[0]+t**3*p3[0]
        y=mt**3*p0[1]+3*mt*mt*t*p1[1]+3*mt*t*t*p2[1]+t**3*p3[1]
        pts.append((x,y))
    return pts

def render(page_idx, form_key, scale=6.0):
    r=pypdf.PdfReader(PDF)
    o=r.pages[page_idx]["/Resources"]["/XObject"][form_key].get_object()
    bbox=[float(v) for v in o["/BBox"]]
    mtx=o.get("/Matrix")
    base=(scale,0,0,-scale,-bbox[0]*scale, bbox[3]*scale)
    W=int((bbox[2]-bbox[0])*scale)+2; H=int((bbox[3]-bbox[1])*scale)+2
    ops=parse(o.get_data())
    ctm=base; stack=[]; fill=0.0; fstack=[]
    cur=[]; subpaths=[]; start=None; pos=None
    polys=[]   # list of (list_of_subpaths, evenodd)
    for op,a in ops:
        if op=='q': stack.append(ctm); fstack.append(fill)
        elif op=='Q':
            if stack: ctm=stack.pop()
            if fstack: fill=fstack.pop()
        elif op=='g' and len(a)>=1: fill=a[-1]
        elif op=='rg' and len(a)>=3: fill=0.299*a[-3]+0.587*a[-2]+0.114*a[-1]
        elif op=='k' and len(a)>=4:
            c,m_,y_,kk=a[-4:]
            fill=0.299*(1-min(1,c+kk))+0.587*(1-min(1,m_+kk))+0.114*(1-min(1,y_+kk))
        elif op in ('sc','scn') and len(a)>=1:
            fill=a[-1] if len(a)==1 else (0.299*a[-3]+0.587*a[-2]+0.114*a[-1] if len(a)>=3 else a[-1])
        elif op=='cm' and len(a)>=6: ctm=mul(tuple(a[-6:]),ctm)
        elif op=='m' and len(a)>=2:
            if cur: subpaths.append(cur)
            pos=(a[-2],a[-1]); start=pos; cur=[app(ctm,*pos)]
        elif op=='l' and len(a)>=2:
            pos=(a[-2],a[-1]); cur.append(app(ctm,*pos))
        elif op=='c' and len(a)>=6:
            p0=pos; p1=(a[-6],a[-5]); p2=(a[-4],a[-3]); p3=(a[-2],a[-1])
            for pt in flatten_cubic(p0,p1,p2,p3): cur.append(app(ctm,*pt))
            pos=p3
        elif op=='v' and len(a)>=4:
            p0=pos; p1=pos; p2=(a[-4],a[-3]); p3=(a[-2],a[-1])
            for pt in flatten_cubic(p0,p1,p2,p3): cur.append(app(ctm,*pt))
            pos=p3
        elif op=='y' and len(a)>=4:
            p0=pos; p1=(a[-4],a[-3]); p2=(a[-2],a[-1]); p3=p2
            for pt in flatten_cubic(p0,p1,p2,p3): cur.append(app(ctm,*pt))
            pos=p3
        elif op=='h':
            if cur and start is not None: cur.append(app(ctm,*start)); pos=start
        elif op=='re' and len(a)>=4:
            x,y,w,hh=a[-4:]
            if cur: subpaths.append(cur)
            cur=[app(ctm,x,y),app(ctm,x+w,y),app(ctm,x+w,y+hh),app(ctm,x,y+hh),app(ctm,x,y)]
            subpaths.append(cur); cur=[]; pos=(x,y); start=pos
        elif op in ('f','F','f*','B','B*','b','b*'):
            if cur: subpaths.append(cur); cur=[]
            if subpaths: polys.append((subpaths, op.endswith('*'), None, fill))
            subpaths=[]
        elif op in ('S','s'):
            # approximate strokes as thin fills: skip (table rules) -> keep as lines
            if cur: subpaths.append(cur); cur=[]
            if subpaths: polys.append((subpaths,False,'stroke',0.0))
            subpaths=[]
        elif op in ('n','W','W*'):
            if op=='n':
                if cur: subpaths.append(cur); cur=[]
                subpaths=[]
    # rasterize: supersample-free scanline, nonzero winding
    img=np.ones((H,W),dtype=np.float32)
    for entry in polys:
        sps=entry[0]; eo=entry[1]; stroke=(entry[2]=='stroke'); col=entry[3]
        edges=[]
        for sp in sps:
            if len(sp)<2: continue
            pts=sp
            if stroke:
                # draw thin lines
                for i in range(len(pts)-1):
                    x0,y0=pts[i]; x1,y1=pts[i+1]
                    n=int(max(abs(x1-x0),abs(y1-y0)))+1
                    for k in range(n+1):
                        xx=int(round(x0+(x1-x0)*k/n)); yy=int(round(y0+(y1-y0)*k/n))
                        if 0<=yy<H and 0<=xx<W: img[yy,xx]=col
                continue
            if pts[0]!=pts[-1]: pts=pts+[pts[0]]
            for i in range(len(pts)-1):
                edges.append((pts[i],pts[i+1]))
        if not edges: continue
        ys=[p[1] for e in edges for p in e]
        y0=max(0,int(min(ys))); y1=min(H-1,int(max(ys))+1)
        for yy in range(y0,y1+1):
            yc=yy+0.5; xs=[]
            for (ax,ay),(bx,by) in edges:
                if (ay<=yc<by) or (by<=yc<ay):
                    t=(yc-ay)/(by-ay); xs.append((ax+(bx-ax)*t, 1 if by>ay else -1))
            if not xs: continue
            xs.sort()
            if eo:
                for i in range(0,len(xs)-1,2):
                    a=int(np.ceil(xs[i][0]-0.5)); b=int(np.floor(xs[i+1][0]-0.5))
                    if b>=a: img[yy,max(0,a):min(W,b+1)]=col
            else:
                w=0
                for i in range(len(xs)-1):
                    w+=xs[i][1]
                    if w!=0:
                        a=int(np.ceil(xs[i][0]-0.5)); b=int(np.floor(xs[i+1][0]-0.5))
                        if b>=a: img[yy,max(0,a):min(W,b+1)]=col
    return img

def save_png(img, path):
    h,w=img.shape
    g=(np.clip(img,0,1)*255).astype(np.uint8)
    raw=b''.join(b'\x00'+g[i].tobytes() for i in range(h))
    def chunk(t,d):
        c=struct.pack('>I',len(d))+t+d
        return c+struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
    png=b'\x89PNG\r\n\x1a\n'
    png+=chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,0,0,0,0))
    png+=chunk(b'IDAT',zlib.compress(raw,6))
    png+=chunk(b'IEND',b'')
    open(path,'wb').write(png)

if __name__=='__main__':
    pi=int(sys.argv[1]); key=sys.argv[2]; sc=float(sys.argv[3]) if len(sys.argv)>3 else 6.0
    img=render(pi,key,sc)
    out=f"paper/imgs/table_p{pi+1}_{key.strip('/')}.png"
    save_png(img,out); print(out, img.shape)
