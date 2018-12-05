from tkinter import *
from tk_tools import *
from os import listdir
from tkinter import messagebox
from tspdata import *
from mip.model import *
from mip.constants import *

def optimize( problemName : str, cnvs ):
    data = TSPData(problemName)
    #messagebox.showinfo('Info', 'instance {} loaded'.format(problemName))
    

    n = data.n
    d = data.d
    print('solving TSP with {} cities'.format(data.n))

    model = Model(solver_name='cbc')

    # binary variables indicating if arc (i,j) is used on the route or not
    x = [ [ model.add_var(
            type=BINARY) 
                for j in range(n) ] 
                for i in range(n) ]

    # continuous variable to prevent subtours: each
    # city will have a different "identifier" in the planned route
    y = [ model.add_var(
        name='y({})'.format(i),
        lb=0.0,
        ub=n) 
            for i in range(n) ]    
    
    # objective funtion: minimize the distance
    model += xsum( d[i][j]*x[i][j]
                    for j in range(n) for i in range(n) )

    # constraint : enter each city coming from another city
    for i in range(n):
        model += xsum( x[j][i] for j in range(n) if j != i ) == 1, 'enter({})'.format(i)
        
    # constraint : leave each city coming from another city
    for i in range(n):
        model += xsum( x[i][j] for j in range(n) if j != i ) == 1, 'leave({})'.format(i)
        
    # subtour elimination
    for i in range(0, n):
        for j in range(0, n):
            if i==j or i==0 or j==0:
                continue
            model += \
                y[i]  - (n+1)*x[i][j] >=  y[j] -n, 'noSub({},{})'.format(i,j)
        
    model.optimize(  )
    
    for i in range(n):
        for j in range(n):
            if x[i][j].x >= 0.98:
                cnvs.create_line( data.ix[i], data.iy[i], data.ix[j], data.iy[j], arrow=LAST, width=3 )
                print('arc {} {}'.format(i,j))


    

w = Tk()
w.title('Trip Planner')
w.geometry( "1100x860")


frame = Frame(w, relief=RAISED, borderwidth=2)
#lbli = Label(w, image=imap)
#lbli.pack(pady=5, side=BOTTOM)


imap = PhotoImage(file='./img/belgium-tourism-14.gif')
cnvs = Canvas(frame, width=800, height=600)


def load_map( mapFile : str ):
    global cnvs, imap
    print('loading {}'.format(mapFile))
    cnvs.delete('all')
    imap = PhotoImage(file=mapFile)
    cnvs.create_image(0,0, anchor=NW, image=imap)
    cnvs.pack(fill=BOTH, expand=1)
    #messagebox.showinfo('ha', 'ha')




frame.pack(fill=BOTH, expand=True, side=BOTTOM)


lbl = Label(w, text='Select instance: ')
lbl.pack( padx=5, side=LEFT )

insts = []
fls = listdir('.')
for f in fls:
    if '.tsp' in f:
        insts.append(f)
        

def selectInstance():
    mfile = './img/{}'.format( selInst.get().replace('.tsp', '.gif') )
    load_map( mfile )
    
selInst = SmartOptionMenu(w, insts)
selInst.add_callback( selectInstance )
selInst.pack( padx=5, side=LEFT)

"""
selInst = StringVar(w)


selInst.set(insts[0]) # set the default option

popupMenu = OptionMenu(w, selInst, *insts, command=selectInstance())
popupMenu.pack(padx=5, side=LEFT)
selInst.trace('w', selectInstance)
"""



lbl2 = Label(w, text='Time limit: (s)' )
lbl2.pack( padx=5, side=LEFT )

tmLim = Entry(w)
tmLim.insert(END, '10')
tmLim.pack( padx=5, side=LEFT )


btnOpt = Button(w, text='Optimize', command=lambda: optimize(selInst.get(), cnvs))
btnOpt.pack( padx=5, side=LEFT)



lblimg=Label()

selectInstance()


w.mainloop()
