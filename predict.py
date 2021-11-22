from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm
from madmom.features.beats import RNNBeatProcessor
from madmom.features.beats import BeatTrackingProcessor
from madmom.features.onsets import OnsetPeakPickingProcessor
from madmom.features.onsets import RNNOnsetProcessor
import librosa
import librosa.display

def make_prediction(args):

    model = load_model(args.model_fn,
        custom_objects={'STFT':STFT,
                        'Magnitude':Magnitude,
                        'ApplyFilterbank':ApplyFilterbank,
                        'MagnitudeToDecibel':MagnitudeToDecibel})
    wav_paths = glob('{}/**'.format(args.src_dir), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
    classes = ['Cymbal', 'Hi_Hat', 'Kick', 'Snare', 'Tom']

    for z, wav_fn in tqdm(enumerate(wav_paths), total=len(wav_paths)):
      print(wav_fn.split("/")[-1])
      results = []
      results_txt = []
      y, sr = librosa.load(wav_fn, sr=args.sr)
      proc = OnsetPeakPickingProcessor(threshold=0.2,fps=100)
      act = RNNOnsetProcessor()(wav_fn)
      np.set_printoptions(suppress=True) 
      onset_samples = proc(act)*args.sr
      starts = onset_samples
      starts = starts.astype(np.int32)
      #starts = np.insert(starts,0,0.02*args.sr)
      #print(starts)
      stops = onset_samples+1600
      stops = stops.astype(np.int32)
      #print(stops)
      output_name = args.pred_fn + '_' + wav_fn.split("/")[-1]
      output_name = output_name.strip('.wav')
      script_dir = os.path.dirname(__file__)
      file_path = os.path.join(script_dir, "logs_multi3/output_txt3_2/"+output_name+".txt")
      f= open(file_path,"w+")
      for j, (start, stop) in enumerate(zip(starts, stops)):
        wav_onset = y[start:stop]
        #rate, wav = downsample_mono(wav_onset, args.sr)
        #mask, env = envelope(wav, rate, threshold=args.threshold)
        #clean_wav = wav[mask]
        step = int(args.sr*args.dt)
        batch = []

        for i in range(0, wav_onset.shape[0], step):
          sample = wav_onset[i:i+step]
          sample = sample.reshape(-1, 1)
          if sample.shape[0] < step:
            tmp = np.zeros(shape=(step, 1), dtype=np.float32)
            tmp[:sample.shape[0],:] = sample.flatten().reshape(-1, 1)
            sample = tmp
          batch.append(sample)
        X_batch = np.array(batch, dtype=np.float32)
        y_pred = model.predict(X_batch)
        y_mean = np.mean(y_pred, axis=0)#改成list
        y_pred_max = np.argmax(y_mean)
        #real_class = os.path.dirname(wav_onset).split('/')[-1]
        #print('Actual class: {}, Predicted class: {}'.format(real_class, classes[y_pred]))
        #print('Predict time: {}, Predicted class: {}'.format(start/args.sr, classes[y_pred]))
        #print(y_pred)
        print('Predict time: ',start/args.sr,'Predicted: ',y_mean)
        #print(classes)
        #print(y_mean)
        #f.write("%.3f, %s\n" % (start/args.sr,classes[y_pred_max]))
        f.write("%.3f %.8f %.8f %.8f %.8f %.8f\n" % (start/args.sr - 1,y_mean[0],y_mean[1],y_mean[2],y_mean[3],y_mean[4]))
        #print(y_pred)
        results.append(y_mean)
        #text = 'Predict time: '+ start/args.sr + 'Predicted class: ' + classes[y_pred] + '\n'
        #results_txt.append(text)
      #print(results)
      np.save(os.path.join('logs_multi3', output_name), np.array(results))
      f.close()
      #np.savetxt(os.path.join('logs', output_npy), np.array(results_txt))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_fn', type=str, default='models/conv2d.multi2',
                        help='model file to make predictions')
    parser.add_argument('--pred_fn', type=str, default='y_pred',
                        help='fn to write predictions in logs dir')
    parser.add_argument('--src_dir', type=str, default='wavfiles_test_whole',
                        help='directory containing wavfiles to predict')
    parser.add_argument('--dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='sample rate of clean audio')
    parser.add_argument('--threshold', type=str, default=20,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

    make_prediction(args)

