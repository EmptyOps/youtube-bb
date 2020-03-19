
########################################################################
# YouTube BoundingBox
########################################################################
#
# This file contains useful functions for downloading, decoding, and
# converting the YouTube BoundingBox dataset.
#
# Author: Mark Buckler
#
########################################################################

from __future__ import unicode_literals
from ffmpy import FFmpeg
from subprocess import check_call
from concurrent import futures
from random import shuffle
from datetime import datetime
import subprocess
import youtube_dl
import socket
import os
import io
import sys
import csv

import pandas as pd
import pickle

# Debug flag. Set this to true if you would like to see ffmpeg errors
debug = False

# The data sets to be downloaded
d_sets = [
          'yt_bb_detection_train',
          'yt_bb_detection_validation',
          'yt_bb_classification_train',
          'yt_bb_classification_validation',
          ]

# The classes included and their indices
class_list = [\
              [0,'person'],
              [1,'bird'],
              [2,'bicycle'],
              [3,'boat'],
              [4,'bus'],
              [5,'bear'],
              [6,'cow'],
              [7,'cat'],
              [8,'giraffe'],
              [9,'potted plant'],
              [10,'horse'],
              [11,'motorcycle'],
              [12,'knife'],
              [13,'airplane'],
              [14,'skateboard'],
              [15,'train'],
              [16,'truck'],
              [17,'zebra'],
              [18,'toilet'],
              [19,'dog'],
              [20,'elephant'],
              [21,'umbrella'],
              [22,'none'],
              [23,'car'],
              ]

# Host location of segment lists
web_host = 'https://research.google.com/youtube-bb/'

# Video clip class
class video_clip(object):
  def __init__(self,
               name,
               yt_id,
               start,
               stop,
               class_id,
               obj_id,
               d_set_dir):
    # name = yt_id+class_id+object_id
    self.name     = name
    self.yt_id    = yt_id
    self.start    = start
    self.stop     = stop
    self.class_id = class_id
    self.obj_id   = obj_id
    self.d_set_dir = d_set_dir
  def print_all(self):
    print('['+self.name+', '+ \
              self.yt_id+', '+ \
              self.start+', '+ \
              self.stop+', '+ \
              self.class_id+', '+ \
              self.obj_id+']\n')

# Video class
class video(object):
  def __init__(self,yt_id,first_clip):
    self.yt_id = yt_id
    self.clips = [first_clip]
  def print_all(self):
    print(self.yt_id)
    for clip in self.clips:
      clip.print_all()

# XML detection annotation class
class xml_annot(object):
  def __init__(self,
               annot_name,
               filename,
               annotation,
               image_width,
               image_height,
               truncated,
               xmin,
               ymin,
               xmax,
               ymax):
    self.annot_name     = annot_name
    self.folder         = "youtubebb2017"
    self.filename       = filename
    self.database       = "YouTube Bounding Box"
    self.annotation     = ",".join(annotation)
    self.image_source   = "YouTube"
    self.image_flickrid = "N/A"
    self.owner_name     = "N/A"
    self.owner_flickrid = "N/A"
    self.image_width    = str(image_width)
    self.image_height   = str(image_height)
    self.image_depth    = str(3)
    self.segmented      = str(0)
    self.class_name     = annotation[3]
    self.pose           = "Unspecified"
    self.truncated      = str(truncated)
    self.difficult      = str(0)
    self.xmin           = str(xmin)
    self.ymin           = str(ymin)
    self.xmax           = str(xmax)
    self.ymax           = str(ymax)


# Download and cut a clip to size
def dl_and_cut(vid):

  d_set_dir = vid.clips[0].d_set_dir

  # Use youtube_dl to download the video
  FNULL = open(os.devnull, 'w')
  check_call(['youtube-dl', \
    #'--no-progress', \
    '-f','best[ext=mp4]', \
    '-o',d_set_dir+'/'+vid.yt_id+'_temp.mp4', \
    'youtu.be/'+vid.yt_id ], \
     stdout=FNULL,stderr=subprocess.STDOUT )

  for clip in vid.clips:
    # Verify that the video has been downloaded. Skip otherwise
    if os.path.exists(d_set_dir+'/'+vid.yt_id+'_temp.mp4'):
      # Make the class directory if it doesn't exist yet
      class_dir = d_set_dir+'/'+str(clip.class_id)
      check_call(' '.join(['mkdir', '-p', class_dir]), shell=True)

      # Cut out the clip within the downloaded video and save the clip
      # in the correct class directory. Full re-encoding is used to maintain
      # frame accuracy. See here for more detail:
      # http://www.markbuckler.com/post/cutting-ffmpeg/
      if debug:
          check_call(['ffmpeg',\
            '-i','file:'+d_set_dir+'/'+vid.yt_id+'_temp.mp4',\
            '-ss', str(float(clip.start)/1000),\
            '-strict','-2',\
            '-t', str((float(clip.stop)-float(clip.start))/1000),\
            '-threads','1',\
            class_dir+'/'+clip.name+'.mp4'])
      else:
          # If not debugging, hide the error outputs from failed downloads
          check_call(['ffmpeg',\
            '-i','file:'+d_set_dir+'/'+vid.yt_id+'_temp.mp4',\
            '-ss', str(float(clip.start)/1000),\
            '-strict','-2',\
            '-t', str((float(clip.stop)-float(clip.start))/1000),\
            '-threads','1',\
            class_dir+'/'+clip.name+'.mp4'],
            stdout=FNULL,stderr=subprocess.STDOUT )

  # Remove the temporary video
  os.remove(d_set_dir+'/'+vid.yt_id+'_temp.mp4')


# Parse the annotation csv file and schedule downloads and cuts
# read_csv_with_alternate_reader: memory savvy reader
def parse_annotations(d_set,dl_dir,rec_ind,offset_min,offset_max,dl_cls_by_filter=-1, read_csv_with_alternate_reader=False):

  rig = 0
  vids = []

  dl_cls_by_filter = str(dl_cls_by_filter)

  pickleFile = 'parse_annotations.pickle'
  pickleObj = {}
  if os.path.exists( pickleFile ):
    with open(pickleFile, 'rb') as handle:
      pickleObj = pickle.load(handle) 

  d_set_dir = dl_dir+'/'+d_set+'/'

  # Download & extract the annotation list
  if not os.path.exists(d_set+'.csv'):
    print (d_set+': Downloading annotations...')
    check_call(' '.join(['wget', web_host+d_set+'.csv.gz']),shell=True)
    print (d_set+': Unzipping annotations...')
    check_call(' '.join(['gzip', '-d', '-f', d_set+'.csv.gz']), shell=True)

  print (d_set+': Parsing annotations into clip data...')

  # Parse csv data.
  if not read_csv_with_alternate_reader:
    annotations = []
    with open((d_set+'.csv'), 'rt') as f:
      reader = csv.reader(f)
      annotations = list(reader)

    # Sort to de-interleave the annotations for easier parsing. We use
    # `int(l[1])` to sort by the timestamps numerically; the other fields are
    # sorted lexicographically as strings.
    print(d_set + ': Sorting annotations...')
    if ('classification' in d_set):
      class_or_det = 'class'
      # Sort by youtube_id, class, and then timestamp
      annotations.sort(key=lambda l: (l[0], l[2], int(l[1])))
    elif ('detection' in d_set):
      class_or_det = 'det'
      # Sort by youtube_id, class, obj_id and then timestamp
      annotations.sort(key=lambda l: (l[0], l[2], l[4], int(l[1])))

    current_clip_name = ['blank']
    clips             = []

    # Parse annotations into list of clips with names, youtube ids, start
    # times and stop times
    for idx, annotation in enumerate(annotations):
      # If this is for a classify dataset there is no object id
      if (class_or_det == 'class'):
        obj_id = '0'
      elif (class_or_det == 'det'):
        obj_id = annotation[4]
      yt_id    = annotation[0]
      class_id = annotation[2]

      #added filter to download only specified class in the filter 
      if not dl_cls_by_filter == -1 and not (int(dl_cls_by_filter)+1) == (int(class_id)+1):
        continue

      clip_name = yt_id+'+'+class_id+'+'+obj_id

      # If this is a new clip
      if clip_name != current_clip_name:

        # Update the finishing clip
        if idx != 0: # If this isnt the first clip
          clips[-1].stop = annotations[idx-1][1]

        # Add the starting clip
        clip_start = annotation[1]
        clips.append( video_clip( \
          clip_name, \
          yt_id, \
          clip_start, \
          '0', \
          class_id, \
          obj_id, \
          d_set_dir) )

        # Update the current clip name
        current_clip_name = clip_name

    # Update the final clip with its stop time
    clips[-1].stop = annotations[-1][1]

    # Sort the clips by youtube id
    clips.sort(key=lambda x: x.yt_id)

    # Create list of videos to download (possibility of multiple clips
    # from one video)
    current_vid_id = ['blank']
    vids = []
    for clip in clips:

      vid_id = clip.yt_id

      # If this is a new video
      if vid_id != current_vid_id:
        # Add the new video with its first clip
        vids.append( video ( \
          clip.yt_id, \
          clip ) )
      # If this is a new clip for the same video
      else:
        # Add the new clip to the video
        vids[-1].clips.append(clip)

      # Update the current video name
      current_vid_id = vid_id
  else:
    # annotations = []
    # with open((d_set+'.csv'), 'rt') as f:
    #   reader = csv.reader(f)
    #   annotations = list(reader)

    df = None 
    chunksize = 2000

    # 
    if ('classification' in d_set):
      class_or_det = 'class'
    elif ('detection' in d_set):
      class_or_det = 'det'

    #create a sorted file first if it isn't exist
    filesorted = d_set+'_'+dl_cls_by_filter+'_sorted.csv'
    if not os.path.exists(filesorted):  
      print(d_set + ': Sorting annotations...')

      if not dl_cls_by_filter == -1:
        # iter_csv = pd.read_csv(d_set+'.csv', iterator=True, chunksize=chunksize)
        # fstr = dl_cls_by_filter   # str(dl_cls_by_filter-1)
        # print( "fstr", fstr )
        # df = pd.concat([chunk[ (chunk[chunk.columns[2]] == fstr) ] for chunk in iter_csv])

        fcnt = -1
        ccnt = -1
        fval = int(dl_cls_by_filter) + 1
        is_header_created = False
        coll = 0
        with open( filesorted, 'wb' ) as file:
          for chunk in pd.read_csv(d_set+'.csv', header=None, dtype=str, chunksize=chunksize):
            # # if True or df is None:
            # if not is_header_created:
            #   # df = pd.DataFrame(columns=chunk.columns) #['lib', 'qty1', 'qty2'])
            #   is_header_created = True
            #   line = ""
            #   print( type(chunk.columns), chunk.columns)
            #   coll = len(chunk.columns)
            #   for colnm in chunk.columns:
            #     line += "\""+colnm+"\","
            #   file.write(line.encode())
            #   file.write('\n'.encode())

            ccnt += 1

            for idx, annotation in chunk.iterrows():
              # print( "columns", chunk.columns, chunk.columns[2] )
              # print( "annotation", annotation )
              if int(annotation[chunk.columns[2]])+1 == fval:
                # fcnt += 1
                # df.loc[fcnt] = annotation
                line = ""
                for colnm in chunk.columns:
                  line += "\""+annotation[colnm]+"\","
                  # print( "colnm ", colnm, annotation[colnm] )
                file.write(line.encode())
                file.write('\n'.encode())

            print("chunk ", ccnt)

        #
        df = pd.read_csv(filesorted, header=None, dtype=str)

        print( "df", len(df.index), fcnt )
      else:
        #load entire csv file in dataframe in df var
        df = pd.read_csv(d_set+'.csv', header=None, dtype=str)

      print("sorting the df frame")

      #
      df = df.astype({df.columns[1]: int})

      #sort 
      # Sort to de-interleave the annotations for easier parsing. We use
      # `int(l[1])` to sort by the timestamps numerically; the other fields are
      # sorted lexicographically as strings.
      if ('classification' in d_set):
        # class_or_det = 'class'
        # Sort by youtube_id, class, and then timestamp
        # annotations.sort(key=lambda l: (l[0], l[2], int(l[1])))
        df.sort_values([df.columns[0], df.columns[2], df.columns[1]], ascending=[True, True, True])
      elif ('detection' in d_set):
        # class_or_det = 'det'
        # Sort by youtube_id, class, obj_id and then timestamp
        # annotations.sort(key=lambda l: (l[0], l[2], l[4], int(l[1])))
        df.sort_values([df.columns[0], df.columns[2], df.columns[4], df.columns[1]], ascending=[True, True, True, True])

      #
      pickleObj[d_set+"_"+dl_cls_by_filter+"_totr"] = len(df.index)

      #write sorted file
      df.to_csv ( filesorted, index = False, header=True )

      #
      with open(pickleFile, 'wb') as handle:
        pickle.dump(pickleObj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
      print(d_set + ': Sorted annotations file found...')

    #
    print("loading sorted csv file and creating clips")

    rig = pickleObj[d_set+"_"+dl_cls_by_filter+"_totr"]

    #create dataframe of only applicable offset range 
    if not offset_min == -1:
      #offset logic, need to move down in clips/vids object loop
      # df = pd.read_csv( filesorted, skiprows=offset_min, nrows=offset_max )

      df = None   #read in chunks instead pd.read_csv(filesorted, header=None, dtype=str)
    else:
      #load entire csv file in dataframe in df var
      df = None   #read in chunks instead pd.read_csv(filesorted, header=None, dtype=str)

    current_clip_name = ['blank']
    current_yt_id = ['blank']
    clips             = []
    vids_fcnt = 0
    vid_pcidx = -1
    ccnt = -1

    # Parse annotations into list of clips with names, youtube ids, start
    # times and stop times
    # for idx, annotation in enumerate(annotations):
    break_loop = False
    for df in pd.read_csv(filesorted, header=None, dtype=str, chunksize=chunksize):
      ccnt += 1
      print("chunk ", ccnt)

      for idx, annotation in df.iterrows():
        
        # If this is for a classify dataset there is no object id
        if (class_or_det == 'class'):
          obj_id = '0'
        elif (class_or_det == 'det'):
          obj_id = annotation[df.columns[4]] #[4]
        yt_id    = annotation[df.columns[0]] #[0]
        class_id = annotation[df.columns[2]] #[2]

        # #added filter to download only specified class in the filter 
        # if not dl_cls_by_filter == -1 and not (int(dl_cls_by_filter)+1) == (int(class_id)+1):
        #   continue

        clip_name = yt_id+'+'+class_id+'+'+obj_id

        # If this is a new clip
        if clip_name != current_clip_name:
          if yt_id != current_yt_id:
            current_yt_id = yt_id
            vids_fcnt += 1

          if vids_fcnt >= offset_min and vids_fcnt <= offset_max:
            vid_pcidx += 1

            # Update the finishing clip
            if vid_pcidx != 0: # idx != 0: # If this isnt the first clip
              clips[-1].stop = df[idx-1:idx][df.columns[1]]  #annotations[idx-1][1]

            # Add the starting clip
            clip_start = annotation[df.columns[1]]  #[1]
            clips.append( video_clip( \
              clip_name, \
              yt_id, \
              clip_start, \
              '0', \
              class_id, \
              obj_id, \
              d_set_dir) )

          elif vids_fcnt > offset_max:
            break_loop = True

          # Update the current clip name
          current_clip_name = clip_name

        if break_loop:
          break

      if break_loop:
        break

    # Update the final clip with its stop time
    if not break_loop:
      lind = len(df.index)
      clips[-1].stop = df[lind-1:lind][df.columns[1]]  #annotations[-1][1]

    #
    print("sorting clips")

    # Sort the clips by youtube id
    clips.sort(key=lambda x: x.yt_id)

    #
    print("looping through clips and creating vid objects")

    # Create list of videos to download (possibility of multiple clips
    # from one video)
    current_vid_id = ['blank']
    for clip in clips:

      vid_id = clip.yt_id

      # If this is a new video
      if vid_id != current_vid_id:
        # Add the new video with its first clip
        vids.append( video ( \
          clip.yt_id, \
          clip ) )
      # If this is a new clip for the same video
      else:
        # Add the new clip to the video
        vids[-1].clips.append(clip)

      # Update the current video name
      current_vid_id = vid_id

  #
  with open(pickleFile, 'wb') as handle:
    pickle.dump(pickleObj, handle, protocol=pickle.HIGHEST_PROTOCOL)

  #return annotations,clips,vids, rig
  return None,None,vids, rig

def sched_downloads(d_set,dl_dir,num_threads,vids,rec_ind=-1,offset_min=-1,offset_max=-1,FREE_SPACE_LIMIT=-1):
  d_set_dir = dl_dir+'/'+d_set+'/'

  # Make the directory for this dataset
  check_call(' '.join(['mkdir', '-p', d_set_dir]), shell=True)

  # Tell the user when downloads were started
  datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  #offset
  if not offset_min == -1:
    vids_tmp = []
    for vid in vids:
      rec_ind = rec_ind + 1
      if rec_ind >= offset_min and rec_ind < offset_max:
        vids_tmp.append(vid)

    vids = vids_tmp

  # Download and cut in parallel threads giving
  with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
    fs = [executor.submit(dl_and_cut,vid) for vid in vids]
    for i, f in enumerate(futures.as_completed(fs)):
      # Write progress to error so that it can be seen
      sys.stderr.write( \
        "Downloaded video: {} / {} \r".format(i, len(vids)))

      if not FREE_SPACE_LIMIT == -1 and get_free_space_in_dir_mount(dl_dir) <= FREE_SPACE_LIMIT:
        prit("free space limit reached. terminating process.")
        exit()

  if len(vids) == 0 and not offset_min == -1:
    print( 'No video found for downloading in dataset '+d_set+'. If this is not intended behaviour, please check whether your offset is set correctly or not' )
  else:
    print( d_set+': All videos downloaded' )

  return rec_ind

def get_free_space_in_dir_mount(path):
  "return free space in mb of the mount where specified path is stored"
  df = subprocess.Popen(["df", path], stdout=subprocess.PIPE)
  output = df.communicate()[0].split()
  # print(output)
  # print(output[10])
  return  float(output[10])/1024.0
  