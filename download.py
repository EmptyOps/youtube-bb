
########################################################################
# YouTube BoundingBox Downloader
########################################################################
#
# This script downloads all videos within the YouTube BoundingBoxes
# dataset and cuts them to the defined clip size. It is accompanied by
# a second script which decodes the videos into single frames.
#
# Author: Mark Buckler
#
########################################################################
#
# The data is placed into the following directory structure:
#
# dl_dir/videos/d_set/class_id/clip_name.mp4
#
########################################################################

import youtube_bb
import sys
from subprocess import check_call

# Parse the annotation csv file and schedule downloads and cuts
def parse_and_sched(dl_dir='videos',num_threads=4,dl_cls_by_filter=-1,offset_min=-1,offset_max=-1,FREE_SPACE_LIMIT=-1):
  """Download the entire youtube-bb data set into `dl_dir`.
  """

  # Make the download directory if it doesn't already exist
  check_call(['mkdir', '-p', dl_dir])

  #
  read_csv_with_alternate_reader=True

  # For each of the four datasets
  rec_ind_glb = 0
  rec_ind = -1
  for d_set in youtube_bb.d_sets:

    offset_min_tmp = offset_min
    offset_max_tmp = offset_max

    if read_csv_with_alternate_reader:
      offset_min_tmp -= rec_ind_glb
      offset_max_tmp -= rec_ind_glb

    annotations,clips,vids,rig = youtube_bb.parse_annotations(d_set,dl_dir,rec_ind,offset_min_tmp,offset_max_tmp,dl_cls_by_filter,read_csv_with_alternate_reader=read_csv_with_alternate_reader)
    
    if read_csv_with_alternate_reader:
      rec_ind_glb += rig
    
      rec_ind = -1
      offset_min_tmp = 0
      offset_max_tmp = offset_max - offset_min

    rec_ind = youtube_bb.sched_downloads(d_set,dl_dir,num_threads,vids,rec_ind,offset_min_tmp,offset_max_tmp,FREE_SPACE_LIMIT)

  print("Processing finished of all datasets")

if __name__ == '__main__':

  assert(len(sys.argv) <= 6), \
          "Usage: python download.py [VIDEO_DIR] [NUM_THREADS] [CLASS_FILTER(optional)] [OFFSET_MIN(optional, offset work with accumulated sum of all 4 datasets)] [OFFSET_MAX(optional, offset work with accumulated sum of all 4 datasets)] [FREE_SPACE_LIMIT(optional, specify in MBs the number of MBs worth of free space that you want to retain on the VIDEO_DIR mount so that it stops when free space go below specified limit)]"
  # Use the directory `videos` in the current working directory by
  # default, or a directory specified on the command line.
  if len(sys.argv) >= 7:
    parse_and_sched(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]))
  elif len(sys.argv) >= 6:
    parse_and_sched(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]))
  elif len(sys.argv) >= 5:
    parse_and_sched(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))
  elif len(sys.argv) >= 4:
    parse_and_sched(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
  else:  
    parse_and_sched(sys.argv[1],int(sys.argv[2]))
