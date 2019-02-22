#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 23:24:08 2018

@author: shanmukha
"""

import cv2,os
import numpy as np
from matplotlib import pyplot as plt


"""converts page to words"""

class preprocess_to_words:
    """getting canny image"""
    def __init__(self,img):
        self.img = cv2.cv2.imread(img,0)
        self.col_img = cv2.cv2.imread(img,1)
        self.img = cv2.cv2.resize(self.img,(0,0),fx=0.75,fy=0.75)
        self.col_img = cv2.cv2.resize(self.col_img,(0,0),fx=0.75,fy=0.75)
        self.edge_img = cv2.cv2.Canny(self.img,200,250)
        self.rows = self.img.shape[0]
        self.cols = self.img.shape[1]
        
    """finding lines"""
    def line_finder(self,left,right):
        line_count_list = [] #no. of pixel rows which have more whites
        line_top_list = []   #top most pixel rows of each text line
        for r in range(0,self.rows - 1):
            count_whites = 0
            for c in range(left,right):
                if self.edge_img[r][c] == 255:
                    count_whites += 1
            line_count_list.append(count_whites)
        r = 0
        while r < len(line_count_list) - 1:
            if line_count_list[r] > 30:
                line_top_list.append(r)
                for r2 in range(r+3,len(line_count_list)-1):
                    if line_count_list[r2] < 8:
                        line_top_list.append(r2)
                        r = r2
                        break
            r += 1
        self.line_top_list = line_top_list
        
    """finding words"""
    def word_finder(self,left,right):
        col_count_list = []
        word_list = []
        for i in range(0,len(self.line_top_list)-1,2):
            col_count_list.append([])
            for c in range(left,right):
                count = 0 
                for r in range(self.line_top_list[i],self.line_top_list[i+1]):
                    if self.edge_img[r][c] == 255:
                        count += 1
                col_count_list[int(i/2)].append(count)
        self.col_count_list = col_count_list
        for r in range(0,len(col_count_list) - 1):
            c = 0
            word_list.append([])
            while c < len(col_count_list[r]) - 5:
                if col_count_list[r][c] + col_count_list[r][c+1] + col_count_list[r][c+2] + col_count_list[r][c+3] < 5:
                    word_list[r].append(left + c)
                    for c2 in range(c+4,len(col_count_list[r])-5):
                        if col_count_list[r][c2] > 1:
                            word_list[r].append(left + c2)
                            c = c2
                            break
                c += 4
            word_list[r].append(right)
        self.word_list = word_list
        word_coord_list = [] #coordinates of word like c1,r1,c2,r2 top left corner and bottom right corner
        for r in range(0,len(self.line_top_list)-2,2):
            for c in range(1,len(self.word_list[int(r/2)])-2,2):
                if self.word_list[int(r/2)][c+1] - self.word_list[int(r/2)][c] > 4:
                    word_coord_list.append(self.word_list[int(r/2)][c] - 3)
                    word_coord_list.append(self.line_top_list[r] - 5)
                    word_coord_list.append(self.word_list[int(r/2)][c+1] + 3)
                    word_coord_list.append(self.line_top_list[r+1] + 2)
        self.word_coord_list = word_coord_list
        print('words are obtained !')
        
    def boxed_lines(self,left,right):
        for r in range(0,len(self.line_top_list) -2,2):            
            cv2.cv2.rectangle(self.img,(left,self.line_top_list[r] - 5),(right,self.line_top_list[r+1]),0,1)
            cv2.cv2.imshow('Image',self.img)
            cv2.cv2.waitKey(0)
            cv2.cv2.destroyAllWindows()
    
    def boxed_words(self):
        for i in range(0,len(self.word_array)-1,4):         
            cv2.cv2.rectangle(self.img_color,(self.word_coord_list[i],self.word_coord_list[i+1]),(self.word_coord_list[i+2],self.word_coord_list[i+3]),0,1)

    def show_image(self):    
        cv2.cv2.imshow('Image',self.col_img)
        cv2.cv2.waitKey(0)
        cv2.cv2.destroyAllWindows()

    def store_words(self,num_words):
        for i in range(0,len(self.word_coord_list),4):
            cropped = self.col_img[self.word_coord_list[i+1]:self.word_coord_list[i+3],self.word_coord_list[i]:self.word_coord_list[i+2]]
            cv2.cv2.imwrite('/home/shanmukha/AnacondaProjects/Spyder_projects/G.E_Hack/images/words/'+str(num_words + i/4)+'.png',cropped)
            #print(cropped) 
    def segment_page_into_words(self):
         self.line_finder(20,int(self.cols/2))	
         self.word_finder(20,int(self.cols/2))
         self.store_words(0)
         num_words = len(os.listdir('/home/shanmukha/AnacondaProjects/Spyder_projects/G.E_Hack/images/words/'))
         self.word_finder(int(self.cols/2),self.cols)
         self.store_words(num_words)             


i = preprocess_to_words('/home/shanmukha/AnacondaProjects/Spyder_projects/G.E_Hack/images/mask-.png')
i.segment_page_into_words()
i.boxed_lines(20,int(i.cols/2))

"""converts words to letters"""

class preprocess_to_letters:
    """Func for rgb to bw"""
    def __init__(self,img):
        self.img = img
        self.img = cv2.cv2.imread(self.img)
        self.img = cv2.cv2.resize(self.img,(0,0),fx=10,fy=5)
        self.bw_img = cv2.cv2.cvtColor(self.img,cv2.cv2.COLOR_BGR2GRAY)
        self.average = np.average(self.bw_img)
        self.rows = self.img.shape[0]
        self.cols = self.img.shape[1]
        #cv2.cv2.imshow('avengers',self.img)
        #cv2.cv2.waitKey(0)
        #cv2.cv2.destroyAllWindows()
            
    """computing threshold"""
    def thresholding(self,thr_val):
        self.t,self.thr_img = cv2.cv2.threshold(self.bw_img,thr_val,255,cv2.cv2.THRESH_BINARY)
        cv2.cv2.imshow('avengers',self.thr_img)
        cv2.cv2.waitKey(0)
        cv2.cv2.destroyAllWindows()
            
    """finding bottom line"""
    def find_bottom_line(self):
        count_list = []
        for r in range(0,self.thr_img.shape[0]-1):
            count_zero = 0
            for c in range(0,self.thr_img.shape[1]-1):
                if self.thr_img[r][c] == 0:
                    count_zero += 1
            count_list.append(count_zero)
        for i in range(len(count_list)-2,0,-1):
            if count_list[i] > int(self.cols/5):
                bottom_line = i
                break
        print(count_list)
        print('bottom line:',count_list[bottom_line],bottom_line)
        return (count_list,bottom_line)
    
    """removing header line"""
    def remove_header_line(self,count_tuple):
        header_row = count_tuple[0].index(max(count_tuple[0]))
        margin = 11
        upper_img = self.thr_img[0:(header_row - margin),:]
        lower_img = self.thr_img[(header_row + margin):count_tuple[1],:]
        self.final_img = np.concatenate((upper_img,lower_img))
#        cv2.cv2.imshow('avengers',self.final_img)
#        cv2.cv2.waitKey(0)
#        cv2.cv2.destroyAllWindows()
    
    """finding intensity in a region"""
    def zeros_in_region(self,count_list,curr_pos,reg):
        count_zeros = sum(count_list[curr_pos - reg:curr_pos + reg])
        return count_zeros
    
    """finds letters"""
    def letter_finder(self,r1,c1,r2,c2):
        count_list = []
        letter_start_list = []
        letter_start_list.append(c1 + 10)
        for c in range(c1,c2):
            count_zeros = 0
            for r in range(r1,r2):
                if self.final_img[r][c] == 0:
                    count_zeros += 1
            count_list.append(count_zeros)
        c = c1 + 80
        while c < len(count_list) - 2:
            if self.zeros_in_region(count_list,c,3) < 2:
                if c + c1 not in letter_start_list:
                    letter_start_list.append(c + c1 + 10)
                    c += 40
            c += 1
        for start,end in zip(letter_start_list[:],letter_start_list[1:]):
            if end - start < 50:
                letter_start_list.remove(start)
        print('letters obtained !')
        print('letter_start_list :',letter_start_list)
        self.letter_start_list = letter_start_list
        self.count_list = count_list
        self.num_letters = len(self.letter_start_list) - 1
        
    """bounding boxes of letters"""
    def draw_boxes(self):
        for i in range(0,len(self.letter_start_list) - 1):
            cv2.cv2.rectangle(self.final_img,(self.letter_start_list[i],0),                                                                                 
                                  (self.letter_start_list[i+1],self.rows),0,1)
#        cv2.cv2.imshow('avengers',self.final_img)
#        cv2.cv2.waitKey(0)
#        cv2.cv2.destroyAllWindows()    
#        
    """resize the image"""
    def image_resize(self,x,y):
        self.img = cv2.cv2.resize(self.img,fx=x,fy=y)
        
    def show_boxed_img(self):
        cv2.cv2.imshow('boxed image',self.final_img)
        cv2.cv2.waitKey(0)
        cv2.cv2.destroyAllWindows()
    
    def show_img(self):
        cv2.cv2.imshow('original image',self.img)
        cv2.cv2.waitKey(0)
        cv2.cv2.destroyAllWindows()
    
    """plotting intensity"""
    def intensity_plot(self):
        x = list(range(len(self.count_list)))
        print(x)
        plt.plot(x,self.count_list)
        plt.show()
        
    """cropping letters from original image"""
    def crop_letters(self,word_index):
        for i in range(len(self.letter_start_list) - 1):
            letter = self.img[0:self.rows,self.letter_start_list[i]:self.letter_start_list[i+1]]
            letter = cv2.cv2.resize(letter,(0,0),fx=0.5,fy=0.5)
            cv2.cv2.imwrite('letters/' + word_index + str(self.letter_start_list[i] + '.png',letter))
    
    def perform_all_funs(self,word_index):
        y = self.find_bottom_line()
        self.remove_header_line(y)
        self.letter_finder(0,0,self.final_img.shape[0],self.cols)
        self.crop_letters(word_index)
        
        
    
        
        
    
        
        
                
        
    
    
    
          
    
      
