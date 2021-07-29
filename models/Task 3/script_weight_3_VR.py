import time
import bpy
import bmesh
import threading

START_FRAME = 0
END_FRAME = 48

obA = bpy.data.objects["pac1"]
KeyNameA = obA.data.shape_keys.name
shapekey_array_A = obA.data.shape_keys.key_blocks

obB = bpy.data.objects["pac2"]
KeyNameB = obB.data.shape_keys.name
shapekey_array_B = obB.data.shape_keys.key_blocks

shape_A_0 = "Target0"
shape_B_0 = bpy.data.objects["shape_name_" + "0"].data.body


fcurve_1 = []
fcurve_2 = []
'''

for fcurve in bpy.data.actions[KeyName1 + "Action"].fcurves:
    
    if fcurve.data_path == 'key_blocks["' + shape_name_1 + '"].value':
        
        fcurve_1 = fcurve

for fcurve in bpy.data.actions[KeyName2 + "Action"].fcurves:
    
    if fcurve.data_path == 'key_blocks["' + shape_name_2 + '"].value':
        
        fcurve_2 = fcurve
'''
        
import os
filepath = bpy.data.filepath
directory = os.path.dirname(filepath)


def runtime():
    
    file = open(directory+"/weight.csv","a")
    
    row = "FRAME" + ";" + shape_A_0 + ";" + shape_B_0  + ";\n"
    file.write(row)
    
    i = START_FRAME
    while(i<=END_FRAME):
        
        bpy.context.scene.frame_current = i
        
        
        time.sleep(0.5)
        
        
        #row = "\n--------------- Frame "+str(i)+"-------------------\n"
        #file.write(row)
        
        #diff = fcurve_1.evaluate(i) - fcurve_2.evaluate(i)
        
        value_A_0 = shapekey_array_A[shape_A_0].value
        value_B_0 = shapekey_array_B[shape_B_0].value


        
        #row = str(fcurve_1.evaluate(i)) + " " + str(fcurve_2.evaluate(i))
        s1 = str("{0:.8f}".format(value_A_0).replace(".",","))
        s2 = str("{0:.8f}".format(value_B_0).replace(".",",")) 
   
        
        row = str(i) + ";" + s1 + ";" + s2 + ";\n"
        file.write(row)
        
        i+=1
        
 
        
    file.close()
    
    
    
my_thread = threading.Thread(target=runtime)
my_thread.start()