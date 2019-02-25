import email
import re
import os
import threading
import pandas as pd
import numpy as np
import mailparser

# Had to change directory of trec07p since it's too big to upload in repo
# base_dir = 'D:/Codes/DataSets/trec07p/'
# base_dir = 'C:/Users/hrman/Downloads/trec07p/trec07p/'
base_dir_processed = 'src/data/'
csv_name = 'processed_emails.csv'
exitFlag = 0


def extract_email_from_file(email_file):
    mail = mailparser.parse_from_file(email_file)
    return cleanhtml(mail.body).replace('\n', ' ').strip()


def load_data_set(index):
    files = pd.read_csv(index, sep=' ', names=['ham_or_spam', 'email_name'])
    files['email_name'] = files['email_name'].apply(lambda x: str(x)[3:])
    files['ham_or_spam'] = files['ham_or_spam'].map({'spam': 1, 'ham': 0})
    files['text'] = ''

    sub_files = np.array_split(files, 5)
    thread_1 = emailThread(1, "email_1", sub_files[0])
    thread_2 = emailThread(2, "email_2", sub_files[1])
    thread_3 = emailThread(3, "email_3", sub_files[2])
    thread_4 = emailThread(4, "email_4", sub_files[3])
    thread_5 = emailThread(5, "email_5", sub_files[4])
    thread_1.start()
    thread_2.start()
    thread_3.start()
    thread_4.start()
    thread_5.start()
    thread_1.join()
    thread_2.join()
    thread_3.join()
    thread_4.join()
    thread_5.join()

    frames = [thread_1.files, thread_2.files,
              thread_3.files, thread_4.files, thread_5.files]
    result = pd.concat(frames)
    result.to_csv(os.path.join(base_dir_processed, csv_name))
    # for index, row in sub_files[0].iterrows():
    #     print(row['email_name'])
    #     try:
    #         # row['text'] = extract_email_from_file(os.path.join(base_dir, row['email_name']))
    #         sub_files[0].at[index, 'text'] = extract_email_from_file(os.path.join(base_dir, row['email_name']))
    #     except:
    #         continue
    # print(files.loc[list(range(0,100))])
    # print(sub_files[0])


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


class emailThread(threading.Thread):
    def __init__(self, thread_id, name, files):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name
        self.files = files

    def run(self):
        print("Starting" + self.name)
        read_emails(self.name, self.files)
        print("Stopping" + self.name)


def read_emails(thread_name, email_frame):
    for index, row in email_frame.iterrows():
        print(row['email_name'])
        try:
            email_frame.at[index, 'text'] = extract_email_from_file(
                os.path.join(base_dir, row['email_name']))
        except:
            email_frame.at[index, 'text'] = ''
            continue

    if exitFlag:
        thread_name.exit()


if __name__ == "__main__":
    load_data_set(os.path.join(base_dir, 'full/index'))
