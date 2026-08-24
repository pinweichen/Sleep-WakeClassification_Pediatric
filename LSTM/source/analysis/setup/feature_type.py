from enum import Enum


class FeatureType(Enum):
    count = "count"
    motion = "motion"
    heart_rate = "heart rate"
    heart_rate_fft1_30 = 'heart_rate_fft1_30'
    cosine = "cosine"
    circadian_model = "circadian model"
    time = "time"
    motion_vmfft9 = "motion_vmfft9"
    motion_vmfft14 = "motion_vmfft14"
    motion_xfft4 = "motion_xfft4"
    motion_xfft9 = "motion_xfft9"
    motion_mpd = "motion_mpd"
    angley = 'angley'
    bfen = 'bfen'
    y_offset_angle = 'y_offset_angle'
    pmaxband = 'pmaxband'
    age = 'age'
    ggir = 'ggir'
    tlbc = 'tlbc'
    biobank = 'biobank'
    motion_vmfft1_15 = 'motion_vmfft1_15'
    motion_xfft1_15 = 'motion_xfft1_15'
    motion_yfft1_15 = 'motion_yfft1_15'
    motion_zfft1_15 = 'motion_zfft1_15'
    
    motion_vmfft1_30 = 'motion_vmfft1_30'
    motion_xfft1_30 = 'motion_xfft1_30'
    motion_yfft1_30 = 'motion_yfft1_30'
    motion_zfft1_30 = 'motion_zfft1_30'
    
    all_offset_angle = 'all_offset_angle'
    motion_xyzfft1_15 = 'motion_xyzfft1_15'
    diagnoses = 'diagnoses'
