# -*- coding: utf-8 -*-

import codecs
import os
import random
import re
import string
import sys
import platform
import time

import icecream


def cut_part(text, max_width):
    result1 = [text[i:i + max_width] for i in range(0, len(text), max_width)]
    result = '\n'.join(result1)
    return result


class Reader:
    def __init__(self, book_name, txt_path):
        self.book_name = book_name
        self.txt_path = txt_path
        self.contents = []  # 章节名称
        self.book = self.split_book_chapter()

    def split_book_chapter(self):
        """
        读取文本文件，并将文本按章节划分
        :return:
        """
        book_content = {}
        with codecs.open(self.txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            chapters = re.split("第[0-9]+章", text)
            for chapter in chapters:
                text = chapter.split("\n")
                title = text[0]
                text = "\n".join(text)
                book_content[title] = text
        self.contents = list(book_content.keys())
        return book_content

    def update_index(self, this_index):
        """
        根据书的章节名称，获取前一章与后一章
        :param this_index:
        :return:
        """
        last_num = this_index - 1
        next_num = this_index + 1
        last_name = self.contents[this_index - 1]
        next_name = self.contents[this_index + 1]
        return last_num, last_name, next_num, next_name

    def start_read(self, chapter_num=None, chapter_name=None):
        """
        开始阅读, 章节名称为空，且数据库中没有历史记录时，从第一章开始
        :param chapter_num: 章节号
        :param chapter_name: 章节名称
        :return:
        """
        if chapter_num is None:
            if chapter_name is None:
                chapter_num = 1
                chapter_name = self.contents[chapter_num - 1]
            chapter_num = self.contents.index(chapter_name) + 1
        chapter_name = self.contents[chapter_num - 1]

        print("开始！当前章节：第{}章".format(chapter_num))
        read_content = "".join(self.book.get(chapter_name, "当前章节不存在"))
        read_content = cut_part(read_content, max_width=40)
        for t in read_content.split('\n'):
            print('{0}{1}{2}'.format(time.time(), 'Support Set and Query Set segmentation completed', ''.join(
                random.sample(string.ascii_letters + string.digits, random.randint(1, 30)))))
            print(t)

        last_chapter_num, last_chapter_name, next_chapter_num, next_chapter_name = self.update_index(chapter_num)
        # 识别命令，获取新的章节号和章节名称
        forward = ""
        while forward not in ["n", "b", "q"]:
            print("继续！当前章节：第{}章".format(num))
            forward = input("n:下一章 b:上一章 q:退出\n")

        if forward in ['q', 'quit']:
            print("退出！已记录章节：第{}章".format(chapter_num))
            sys.exit(0)
        # 根据命令选择上一章或者下一章，并更新历史记录
        if forward in ['b', 'back']:
            return last_chapter_num, last_chapter_name
        if forward in ['n', 'next']:
            return next_chapter_num, next_chapter_name


if __name__ == '__main__':
    if platform.system().lower() == "windows":
        os.system('cls')
    if platform.system().lower() == "linux":
        os.system('clear')

    mybook = Reader("fwy_name", "fwy.txt")
    num, name = 108, None
    while True:
        num, name = mybook.start_read(chapter_num=num, chapter_name=name)
