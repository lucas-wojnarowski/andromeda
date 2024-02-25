import sys
from tkinter import *
from PIL import ImageTk, Image
import serial
from threading import Thread
import threading
import struct
import pandas as pd
import joblib
import numpy as np
import traceback
import os
import time

global df, df_adc, df_baseline
global key
global greyscale_grid_color
global color, previous_colors, baseline_calibration_done, previous_raw, display_debounce_counter, previous_delta

images_tags_list = [''] * 81
images_references_list = [''] * 81
images_text_reference_list = [''] * 81
FirstStoneX = 157
FirstStoneY = 50
PixelIncrementX = 37
PixelIncrementY = 36
PixelOffsetY = [0, 0, 0, 0, 0, 1, 2, 2, 3]
PixelOffsetX = [1, 0, 0, 0, 0, 0, 0, 0, 0]

port = serial.Serial('COM7', baudrate=117650, timeout=3.0)
root = Tk()
root.title("9x9 Board - Andromeda")
root.geometry("+100+50")

frame = Frame(root)
frame.pack()

canvas = Canvas(frame, width=600, height=400, bg="white")
canvas.pack()

board = Image.open("9x9_board.png").convert("RGBA").crop([0, 55, 500, 500])
background_img = ImageTk.PhotoImage(board)

display_debounce_counter = list(np.zeros(81))
previous_delta = list(np.zeros(81))
previous_raw = list(np.zeros(81))
previous_stones = ["none" for x in range(81)]
previous_colors = list(np.zeros(81))

canvas.create_image(300, 220, image=background_img)

df = pd.DataFrame()
df_adc = pd.DataFrame()
df_baseline = pd.DataFrame()
df_readings = pd.DataFrame()
label_colors = ["n" for x in range(81)]
baseline_calibration_done = False

key = ''
greyscale_grid_color = 0

gs = joblib.load("ADC_to_color.pkl")
gs_2 = joblib.load("color_to_class.pkl")

exit_flag = threading.Event()


def key_input():
    global key
    global greyscale_grid_color

    while not exit_flag.is_set():
        try:
            if key != 's':
                print(f"Measurement nr: {greyscale_grid_color}")
                key = input("Press s to start: ")
            elif key != 'C':
                key = input("Press d to stop: ")
                greyscale_grid_color = greyscale_grid_color + 1
            else:
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            # Handle the KeyboardInterrupt (Ctrl+C) here
            print("Program was interrupted.")
        finally:
            pass
            # Optionally, perform any cleanup here
            # print("Exiting the program.")


def create_stone(file, x, y):
    images_references_list[9 * x + y] = PhotoImage(file=file).subsample(5, 5)
    images_tags_list[9 * x + y] = \
        canvas.create_image(FirstStoneX + PixelIncrementX * y + PixelOffsetX[y],
                            FirstStoneY + PixelIncrementY * x + PixelOffsetY[x],
                            image=images_references_list[9 * x + y])


def remove_stone(x, y):
    canvas.delete(images_tags_list[9 * x + y])
    images_tags_list[9 * x + y] = ''
    images_references_list[9 * x + y] = ''


def convert_uint16_to_int(byte_pair):
    return int.from_bytes(byte_pair, byteorder='big')


def serial_read():
    global df, df_adc, df_baseline, df_readings
    global key
    global greyscale_grid_color
    global color
    global previous_colors, baseline_calibration_done, display_debounce_counter, previous_delta, previous_raw

    while not exit_flag.is_set():
        header = False
        try:
            byte = convert_uint16_to_int(port.read(1))
            if byte == 1:
                byte = convert_uint16_to_int(port.read(1))
                if byte == 0:
                    byte = convert_uint16_to_int(port.read(1))
                    if byte == 162:
                        header = True
                        rcv_adc = port.read(162)
            if header:
                try:
                    adc_int = list(struct.unpack('>81H', rcv_adc))
                    if key == 's':
                        columns = [str(i) for i in range(81)] + [str(f"color_{i}") for i in range(81)]
                        # columns.append('greyscale_value')
                        # data = (df_baseline - df.tail(6).median()).tolist()
                        data = previous_colors + label_colors
                        df_adc_sample = pd.DataFrame([data], columns=columns)
                        df_readings = pd.concat([df_readings, df_adc_sample], ignore_index=True)
                        key = 'd'
                except Exception as e:
                    return

                columns = [str(i) for i in range(81)]
                df_adc = pd.DataFrame([adc_int], columns=columns)
                df = pd.concat([df, df_adc], ignore_index=True)

                for x in range(9):
                    for y in range(9):
                        canvas.delete(images_text_reference_list[9 * x + y])
                        if baseline_calibration_done:
                            delta = df_baseline.iloc[9 * x + y] - df.tail(filter_value)[f"{9 * x + y}"].median()
                            if ((np.abs(delta - previous_delta[9 * x + y]) < 10) &
                                    (display_debounce_counter[9 * x + y] < display_debounce_counter_value)):
                                display_debounce_counter[9 * x + y] += 1
                                color = gs.predict(pd.DataFrame({'0': [delta]})).astype(int)[0][0]
                                stone_class = gs_2.predict(pd.DataFrame({'0': [color]}))
                                if stone_class == 'white' and previous_stones[9 * x + y] != 'white':
                                    create_stone("white_stone.png", x, y)
                                    # print(f"Added white stone at {x}, {y} with color {color}")
                                elif stone_class == 'black' and previous_stones[9 * x + y] != 'black':
                                    create_stone("black_stone.png", x, y)
                                    # print(f"Added black stone at {x}, {y} with color {color}")
                                elif stone_class == 'none' and images_tags_list[9 * x + y] != '':
                                    # print(f"Removed at {x}, {y} with color {color}")
                                    remove_stone(x, y)
                                previous_stones[9 * x + y] = stone_class
                                previous_colors[9 * x + y] = color
                            elif np.abs(delta - previous_delta[9 * x + y]) > 10:
                                display_debounce_counter[9 * x + y] = 0

                            previous_raw[9 * x + y] = adc_int[9 * x + y]
                            previous_delta[9 * x + y] = delta

                            # Colors
                            if color_display:
                                images_text_reference_list[9 * x + y] = \
                                    canvas.create_text(FirstStoneX + PixelIncrementX * y + PixelOffsetX[y],
                                                       FirstStoneY + PixelIncrementY * x + PixelOffsetY[x],
                                                       fill="blue", font="Times 8 italic bold", text=str(color))

                            # Baseline
                            if baseline_display:
                                images_text_reference_list[9 * x + y] = \
                                    canvas.create_text(FirstStoneX + PixelIncrementX * y + PixelOffsetX[y],
                                                       FirstStoneY + PixelIncrementY * x + PixelOffsetY[x],
                                                       fill="blue", font="Times 8 italic bold",
                                                       text=str(df_baseline[f"{9 * x + y}"]))

                            # Delta
                            if delta_display:
                                images_text_reference_list[9 * x + y] = \
                                    canvas.create_text(FirstStoneX + PixelIncrementX * y + PixelOffsetX[y],
                                                       FirstStoneY + PixelIncrementY * x + PixelOffsetY[x],
                                                       fill="blue", font="Times 8 italic bold",
                                                       text=str(delta))

                            # ADC raw
                            if raw_display:
                                images_text_reference_list[9 * x + y] = \
                                    canvas.create_text(FirstStoneX + PixelIncrementX * y + PixelOffsetX[y],
                                                       FirstStoneY + PixelIncrementY * x + PixelOffsetY[x],
                                                       fill="blue", font="Times 8 italic bold",
                                                       text=str(adc_int[9 * x + y]))

                        # Baseline calibration values
                        if x == 8 and y == 8 and df.shape[0] == 3 and not baseline_calibration_done:
                            df_baseline = df.median()
                            baseline_calibration_done = True
                            port.flush()


        except Exception as e:
            traceback.print_exc()
            return


color_display = False
baseline_display = False
delta_display = False
raw_display = False
filter_value = 5
display_debounce_counter_value = 2

t1 = Thread(target=serial_read)
t2 = Thread(target=key_input)
t1.start()
t2.start()
root.mainloop()
df_readings.to_csv("stone_classification_4_sensors_4.csv")
port.close()
exit_flag.set()
t1.join(0.1)

t2.join(0.1)
os._exit(0)
