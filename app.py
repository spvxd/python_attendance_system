import secrets
from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify, flash, current_app
import mysql.connector
import cv2
from PIL import Image
import numpy as np
import os
import time
from datetime import date
import winsound
from playsound import playsound
from secrets import token_bytes

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

cnt = 0
pause_cnt = 0
justscanned = False

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="flask_db",
    charset='utf8'
)

mycursor = mydb.cursor(buffered=True)

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Генерируем датасет >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(nbr):
    face_classifier = cv2.CascadeClassifier(
        "C:/Users/Admin/PycharmProjects/python_attendance_system/resources/haarcascade_frontalface_default.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor=1.3
        # Minimum neighbor = 5

        if faces == ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face

    cap = cv2.VideoCapture(0)

    mycursor.execute("select ifnull(max(img_id), 0) from img_dataset")
    row = mycursor.fetchone()
    lastid = row[0]

    img_id = lastid
    max_imgid = img_id + 100
    count_img = 0

    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            img_id += 1
            if count_img == 5:
                current_path = os.path.dirname(__file__)
                sound_folder = os.path.join(current_path, 'resources/')
                sound = os.path.join(sound_folder, 'wait.wav')
                winsound.PlaySound(sound, winsound.SND_ASYNC)
            if count_img == 100:
                current_path = os.path.dirname(__file__)
                sound_folder = os.path.join(current_path, 'resources/')
                sound = os.path.join(sound_folder, 'ready.wav')
                winsound.PlaySound(sound, winsound.SND_ASYNC)
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = "dataset/" + nbr + "." + str(img_id) + ".jpg"
            cv2.imwrite(file_name_path, face)
            #cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            mycursor.execute  ("""INSERT INTO `img_dataset` (`img_id`, `img_person`) VALUES
                                ('{}', '{}')""".format(img_id, nbr))
            mydb.commit()

            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
                break
                cap.release()
                cv2.destroyAllWindows()


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Обучение классификатора >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/train_classifier/<nbr>')
def train_classifier(nbr):
    dataset_dir = "C:/Users/Admin/PycharmProjects/python_attendance_system/dataset"

    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L');
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)

    # Тренировка классификатора и сохранение
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")

    return redirect('/')


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<     <<<<<<<<<<<<<<<<< Распознавание лица  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def face_recognition():  # генерирование кадра за кадром с камеры
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

        global justscanned
        global pause_cnt

        pause_cnt += 1

        coords = []

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))

            if confidence > 70 and not justscanned:
                global cnt
                cnt += 1

                n = (100 / 30) * cnt
                # w_filled = (n / 100) * w
                w_filled = (cnt / 30) * w

                cv2.putText(img, str(int(n)) + ' %', (x + 20, y + h + 28), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (153, 255, 255), 2, cv2.LINE_AA)
                if n == 10:
                    try:
                        current_path = os.path.dirname(__file__)
                        sound_folder = os.path.join(current_path, 'resources/')
                        sound = os.path.join(sound_folder, 'scaner.wav')
                        winsound.PlaySound(sound, winsound.SND_ASYNC)
                        if n > 90:
                            break
                    except:
                        break

                cv2.rectangle(img, (x, y + h + 40), (x + w, y + h + 50), color, 2)
                cv2.rectangle(img, (x, y + h + 40), (x + int(w_filled), y + h + 50), (153, 255, 255), cv2.FILLED)
                mycursor.execute("select a.img_person, b.prs_name, b.prs_grp "
                                 "  from img_dataset a "
                                 "  left join prs_mstr b on a.img_person = b.prs_nbr "
                                 " where img_id = " + str(id))
                row = mycursor.fetchone()
                pnbr = row[0]
                pname = row[1]
                pgrp = row[2]


                if int(cnt) == 30:
                    cnt = 0

                    mycursor.execute("insert into accs_hist (accs_date, accs_prsn) values('" + str(
                        date.today()) + "', '" + pnbr + "')")
                    mydb.commit()

                    cv2.putText(img, pname + ' | ' + pgrp, (x - 100, y - 30), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (153, 255, 255), 2, cv2.LINE_AA)
                    current_path = os.path.dirname(__file__)
                    sound_folder = os.path.join(current_path, 'resources/')
                    sound = os.path.join(sound_folder, 'beep.wav')
                    winsound.PlaySound(sound, winsound.SND_ASYNC)
                    # time.sleep(0.5)

                    justscanned = True
                    pause_cnt = 0

            else:
                if not justscanned:
                    cv2.putText(img, 'Неизвестный', (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(img, '', (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                if pause_cnt > 80:
                    justscanned = False

            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 0), "Face", clf)
        return img

    faceCascade = cv2.CascadeClassifier(
        "C:/Users/Admin/PycharmProjects/python_attendance_system/resources/haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    wCam, hCam = 400, 400

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        ret, img = cap.read()
        img = recognize(img, clf, faceCascade)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        key = cv2.waitKey(1)
        if key == 27:
            break


@app.route('/')
def home():
    mycursor.execute("select prs_nbr, prs_name, prs_grp, prs_course, prs_added from prs_mstr")
    data = mycursor.fetchall()

    return render_template('index.html', data=data)


def save_images(photo):
    hash_photo = secrets.token_urlsafe(10)
    _, file_extension = os.path.splitext(photo.filename)
    photo_name = hash_photo + file_extension
    file_path = os.path.join(current_app.root_path, 'static/images', photo_name)
    photo.save(file_path)
    return photo_name


@app.route('/person/<int:id>')
def person(id):
    mycursor.execute("select * from prs_mstr WHERE prs_nbr=%s", (id,))
    data = mycursor.fetchall()
    mycursor.execute("select accs_added from accs_hist WHERE accs_prsn=%s", (id,))
    # mycursor.execute(
    #     "select accs_id, accs_added, accs_prsn from accs_hist")
    data2 = mycursor.fetchall()

    return render_template('person.html', data=data, data2=data2)


@app.route('/test')
def test():
    return render_template('temp.html')


@app.route('/<int:id>/del')
def delete(id):
    mycursor.execute("DELETE FROM prs_mstr WHERE prs_nbr=%s", (id,))
    mydb.commit()


    mycursor.execute("select prs_nbr, prs_name, prs_grp, prs_course, prs_added from prs_mstr")
    data = mycursor.fetchall()

    return render_template('index.html', data=data)

@app.route('/addprsn')
def addprsn():
    mycursor.execute("select ifnull(max(prs_nbr) + 1, 100) from prs_mstr")
    row = mycursor.fetchone()
    nbr = row[0]
    # print(int(nbr))

    return render_template('temp.html', newnbr=int(nbr))


@app.route('/addprsn_submit', methods=['POST'])
def addprsn_submit():
    prsnbr = request.form.get('txtnbr')
    prsname = request.form.get('txtname')
    prsgrp = request.form.get('optgrp')
    prstel = request.form.get('tel')
    prscourse = request.form.get('course')
    prsemail = request.form.get('email')
    img = save_images(request.files['image'])

    mycursor.execute("""INSERT INTO `prs_mstr` (`prs_nbr`, `prs_name`, `prs_grp`, `prs_tel`, `prs_course`,`prs_email`,`prs_img` ) VALUES
                    ('{}', '{}', '{}', '{}', '{}', '{}', '{}')""".format(prsnbr, prsname, prsgrp, prstel, prscourse, prsemail, img))
    mydb.commit()

    # return redirect(url_for('home'))
    return redirect(url_for('vfdataset_page', prs=prsnbr))


@app.route('/vfdataset_page/<prs>')
def vfdataset_page(prs):
    return render_template('gendataset.html', prs=prs)


@app.route('/vidfeed_dataset/<nbr>')
def vidfeed_dataset(nbr):
    # Поток видео
    return Response(generate_dataset(nbr), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    #Поток видео
    return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/fr_page')
def fr_page():
    """Страница для сканирования."""
    mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.prs_grp, a.accs_added "
                     "  from accs_hist a "
                     "  left join prs_mstr b on a.accs_prsn = b.prs_nbr "
                     " where a.accs_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()

    return render_template('fr_page.html', data=data)

@app.route('/fr_page_delete')
def fr_page_delete():
    mycursor.execute("TRUNCATE TABLE `accs_hist`")
    mydb.commit()
    return render_template('fr_page.html')
@app.route('/countTodayScan')
def countTodayScan():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="flask_db",
        charset='utf8'
    )
    mycursor = mydb.cursor()

    mycursor.execute("select count(*) "
                     "  from accs_hist "
                     " where accs_date = curdate() ")
    row = mycursor.fetchone()
    rowcount = row[0]

    return jsonify({'rowcount': rowcount})


@app.route('/loadData', methods=['GET', 'POST'])
def loadData():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="flask_db"
    )
    mycursor = mydb.cursor()

    mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.prs_grp, date_format(a.accs_added, '%H:%i:%s') "
                     "  from accs_hist a "
                     "  left join prs_mstr b on a.accs_prsn = b.prs_nbr "
                     " where a.accs_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()

    return jsonify(response=data)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
