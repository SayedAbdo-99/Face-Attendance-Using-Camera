import mysql.connector

def addAttendance(camNum,name, attDate):
    mydb = mysql.connector.connect(host="localhost",
                               user="root",
                               passwd="Sayed;99",
                               database="attendance")
    
    mycursor = mydb.cursor()
    sql ="INSERT INTO `cam_"+str(camNum)+"` ( `name`, `attDate`) VALUES ( %s, %s)"
    val = (name, attDate)
    mycursor.execute(sql, val)
    mydb.commit()
    mydb.close()

   # print(mycursor.rowcount, "record inserted.")


#addAttendance("Ali",get_current_time(),d,100)