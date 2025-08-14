

#---------------------------------
# New invocation of recon-all Tue May  7 10:49:47 CDT 2019 

 mri_convert /home/surly-raid1/kendrick-data/nsd/ppdata/NSD134-structurals/T1_0pt8_masked.nii.gz /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/mri/orig/001.mgz 

#--------------------------------------------
#@# MotionCor Tue May  7 10:49:55 CDT 2019

 cp /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/mri/orig/001.mgz /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/mri/rawavg.mgz 


 mri_convert /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/mri/rawavg.mgz /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/mri/orig.mgz --conform_min 


 mri_add_xform_to_header -c /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/mri/transforms/talairach.xfm /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/mri/orig.mgz /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/mri/orig.mgz 

#--------------------------------------------
#@# Talairach Tue May  7 10:50:14 CDT 2019

 mri_nu_correct.mni --no-rescale --i orig.mgz --o orig_nu.mgz --n 1 --proto-iters 1000 --distance 50 


 talairach_avi --i orig_nu.mgz --xfm transforms/talairach.auto.xfm 

talairach_avi log file is transforms/talairach_avi.log...

 cp transforms/talairach.auto.xfm transforms/talairach.xfm 

#--------------------------------------------
#@# Talairach Failure Detection Tue May  7 10:53:00 CDT 2019

 talairach_afd -T 0.005 -xfm transforms/talairach.xfm 


 awk -f /home/stone/software/freesurfer_stable_600/bin/extract_talairach_avi_QA.awk /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/mri/transforms/talairach_avi.log 


 tal_QC_AZS /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/mri/transforms/talairach_avi.log 

#--------------------------------------------
#@# Nu Intensity Correction Tue May  7 10:53:00 CDT 2019

 mri_nu_correct.mni --i orig.mgz --o nu.mgz --uchar transforms/talairach.xfm --cm --n 2 


 mri_add_xform_to_header -c /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/mri/transforms/talairach.xfm nu.mgz nu.mgz 

#--------------------------------------------
#@# Intensity Normalization Tue May  7 10:57:01 CDT 2019

 mri_normalize -g 1 -mprage -noconform nu.mgz T1.mgz 

#--------------------------------------------
#@# Skull Stripping Tue May  7 11:00:46 CDT 2019

 mri_em_register -rusage /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/touch/rusage.mri_em_register.skull.dat -skull nu.mgz /home/stone/software/freesurfer_stable_600/average/RB_all_withskull_2016-05-10.vc700.gca transforms/talairach_with_skull.lta 


 mri_watershed -rusage /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/touch/rusage.mri_watershed.dat -T1 -brain_atlas /home/stone/software/freesurfer_stable_600/average/RB_all_withskull_2016-05-10.vc700.gca transforms/talairach_with_skull.lta T1.mgz brainmask.auto.mgz 


 cp brainmask.auto.mgz brainmask.mgz 

#-------------------------------------
#@# EM Registration Tue May  7 23:10:01 CDT 2019

 mri_em_register -rusage /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/touch/rusage.mri_em_register.dat -uns 3 -mask brainmask.mgz nu.mgz /home/stone/software/freesurfer_stable_600/average/RB_all_2016-05-10.vc700.gca transforms/talairach.lta 

#--------------------------------------
#@# CA Normalize Wed May  8 12:50:28 CDT 2019

 mri_ca_normalize -c ctrl_pts.mgz -mask brainmask.mgz nu.mgz /home/stone/software/freesurfer_stable_600/average/RB_all_2016-05-10.vc700.gca transforms/talairach.lta norm.mgz 

#--------------------------------------
#@# CA Reg Wed May  8 12:55:01 CDT 2019

 mri_ca_register -rusage /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/touch/rusage.mri_ca_register.dat -nobigventricles -T transforms/talairach.lta -align-after -mask brainmask.mgz norm.mgz /home/stone/software/freesurfer_stable_600/average/RB_all_2016-05-10.vc700.gca transforms/talairach.m3z 

#--------------------------------------
#@# SubCort Seg Wed May  8 15:57:43 CDT 2019

 mri_ca_label -relabel_unlikely 9 .3 -prior 0.5 -align norm.mgz transforms/talairach.m3z /home/stone/software/freesurfer_stable_600/average/RB_all_2016-05-10.vc700.gca aseg.auto_noCCseg.mgz 


 mri_cc -aseg aseg.auto_noCCseg.mgz -o aseg.auto.mgz -lta /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/mri/transforms/cc_up.lta FS6EXPERT_subj01 

#--------------------------------------
#@# Merge ASeg Wed May  8 18:11:40 CDT 2019

 cp aseg.auto.mgz aseg.presurf.mgz 

#--------------------------------------------
#@# Intensity Normalization2 Wed May  8 18:11:40 CDT 2019

 mri_normalize -mprage -noconform -aseg aseg.presurf.mgz -mask brainmask.mgz norm.mgz brain.mgz 

#--------------------------------------------
#@# Mask BFS Wed May  8 18:18:34 CDT 2019

 mri_mask -T 5 brain.mgz brainmask.mgz brain.finalsurfs.mgz 

#--------------------------------------------
#@# WM Segmentation Wed May  8 18:18:36 CDT 2019

 mri_segment -mprage brain.mgz wm.seg.mgz 


 mri_edit_wm_with_aseg -keep-in wm.seg.mgz brain.mgz aseg.presurf.mgz wm.asegedit.mgz 


 mri_pretess wm.asegedit.mgz wm norm.mgz wm.mgz 

#--------------------------------------------
#@# Fill Wed May  8 18:22:39 CDT 2019

 mri_fill -a ../scripts/ponscc.cut.log -xform transforms/talairach.lta -segmentation aseg.auto_noCCseg.mgz wm.mgz filled.mgz 

#--------------------------------------------
#@# Tessellate lh Wed May  8 18:24:02 CDT 2019

 mri_pretess ../mri/filled.mgz 255 ../mri/norm.mgz ../mri/filled-pretess255.mgz 


 mri_tessellate ../mri/filled-pretess255.mgz 255 ../surf/lh.orig.nofix 


 rm -f ../mri/filled-pretess255.mgz 


 mris_extract_main_component ../surf/lh.orig.nofix ../surf/lh.orig.nofix 

#--------------------------------------------
#@# Tessellate rh Wed May  8 18:24:11 CDT 2019

 mri_pretess ../mri/filled.mgz 127 ../mri/norm.mgz ../mri/filled-pretess127.mgz 


 mri_tessellate ../mri/filled-pretess127.mgz 127 ../surf/rh.orig.nofix 


 rm -f ../mri/filled-pretess127.mgz 


 mris_extract_main_component ../surf/rh.orig.nofix ../surf/rh.orig.nofix 

#--------------------------------------------
#@# Smooth1 lh Wed May  8 18:24:21 CDT 2019

 mris_smooth -nw -seed 1234 ../surf/lh.orig.nofix ../surf/lh.smoothwm.nofix 

#--------------------------------------------
#@# Smooth1 rh Wed May  8 18:24:31 CDT 2019

 mris_smooth -nw -seed 1234 ../surf/rh.orig.nofix ../surf/rh.smoothwm.nofix 

#--------------------------------------------
#@# Inflation1 lh Wed May  8 18:24:39 CDT 2019

 mris_inflate -no-save-sulc -n 50 ../surf/lh.smoothwm.nofix ../surf/lh.inflated.nofix 

#--------------------------------------------
#@# Inflation1 rh Wed May  8 18:27:02 CDT 2019

 mris_inflate -no-save-sulc -n 50 ../surf/rh.smoothwm.nofix ../surf/rh.inflated.nofix 

#--------------------------------------------
#@# QSphere lh Wed May  8 18:29:23 CDT 2019

 mris_sphere -q -seed 1234 ../surf/lh.inflated.nofix ../surf/lh.qsphere.nofix 

#--------------------------------------------
#@# QSphere rh Wed May  8 18:32:29 CDT 2019

 mris_sphere -q -seed 1234 ../surf/rh.inflated.nofix ../surf/rh.qsphere.nofix 

#--------------------------------------------
#@# Fix Topology Copy lh Wed May  8 18:35:26 CDT 2019

 cp ../surf/lh.orig.nofix ../surf/lh.orig 


 cp ../surf/lh.inflated.nofix ../surf/lh.inflated 

#--------------------------------------------
#@# Fix Topology Copy rh Wed May  8 18:35:26 CDT 2019

 cp ../surf/rh.orig.nofix ../surf/rh.orig 


 cp ../surf/rh.inflated.nofix ../surf/rh.inflated 

#@# Fix Topology lh Wed May  8 18:35:26 CDT 2019

 mris_fix_topology -rusage /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/touch/rusage.mris_fix_topology.lh.dat -mgz -sphere qsphere.nofix -ga -seed 1234 FS6EXPERT_subj01 lh 

#@# Fix Topology rh Wed May  8 19:09:07 CDT 2019

 mris_fix_topology -rusage /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/touch/rusage.mris_fix_topology.rh.dat -mgz -sphere qsphere.nofix -ga -seed 1234 FS6EXPERT_subj01 rh 


 mris_euler_number ../surf/lh.orig 


 mris_euler_number ../surf/rh.orig 


 mris_remove_intersection ../surf/lh.orig ../surf/lh.orig 


 rm ../surf/lh.inflated 


 mris_remove_intersection ../surf/rh.orig ../surf/rh.orig 


 rm ../surf/rh.inflated 

#--------------------------------------------
#@# Make White Surf lh Wed May  8 19:36:26 CDT 2019

 mris_make_surfaces -aseg ../mri/aseg.presurf -white white.preaparc -noaparc -whiteonly -mgz -T1 brain.finalsurfs FS6EXPERT_subj01 lh 

#--------------------------------------------
#@# Make White Surf rh Wed May  8 19:44:18 CDT 2019

 mris_make_surfaces -aseg ../mri/aseg.presurf -white white.preaparc -noaparc -whiteonly -mgz -T1 brain.finalsurfs FS6EXPERT_subj01 rh 

#--------------------------------------------
#@# Smooth2 lh Wed May  8 19:49:59 CDT 2019

 mris_smooth -n 3 -nw -seed 1234 ../surf/lh.white.preaparc ../surf/lh.smoothwm 

#--------------------------------------------
#@# Smooth2 rh Wed May  8 19:50:08 CDT 2019

 mris_smooth -n 3 -nw -seed 1234 ../surf/rh.white.preaparc ../surf/rh.smoothwm 

#--------------------------------------------
#@# Inflation2 lh Wed May  8 19:50:18 CDT 2019

 mris_inflate -rusage /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/touch/rusage.mris_inflate.lh.dat -n 50 ../surf/lh.smoothwm ../surf/lh.inflated 

#--------------------------------------------
#@# Inflation2 rh Wed May  8 19:51:28 CDT 2019

 mris_inflate -rusage /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/touch/rusage.mris_inflate.rh.dat -n 50 ../surf/rh.smoothwm ../surf/rh.inflated 

#--------------------------------------------
#@# Curv .H and .K lh Wed May  8 19:52:31 CDT 2019

 mris_curvature -w lh.white.preaparc 


 mris_curvature -thresh .999 -n -a 5 -w -distances 10 10 lh.inflated 

#--------------------------------------------
#@# Curv .H and .K rh Wed May  8 19:54:23 CDT 2019

 mris_curvature -w rh.white.preaparc 


 mris_curvature -thresh .999 -n -a 5 -w -distances 10 10 rh.inflated 


#-----------------------------------------
#@# Curvature Stats lh Wed May  8 19:56:11 CDT 2019

 mris_curvature_stats -m --writeCurvatureFiles -G -o ../stats/lh.curv.stats -F smoothwm FS6EXPERT_subj01 lh curv sulc 


#-----------------------------------------
#@# Curvature Stats rh Wed May  8 19:56:17 CDT 2019

 mris_curvature_stats -m --writeCurvatureFiles -G -o ../stats/rh.curv.stats -F smoothwm FS6EXPERT_subj01 rh curv sulc 

#--------------------------------------------
#@# Sphere lh Wed May  8 19:56:23 CDT 2019

 mris_sphere -rusage /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/touch/rusage.mris_sphere.lh.dat -seed 1234 ../surf/lh.inflated ../surf/lh.sphere 

#--------------------------------------------
#@# Sphere rh Wed May  8 20:09:34 CDT 2019

 mris_sphere -rusage /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/touch/rusage.mris_sphere.rh.dat -seed 1234 ../surf/rh.inflated ../surf/rh.sphere 

#--------------------------------------------
#@# Surf Reg lh Wed May  8 20:22:33 CDT 2019

 mris_register -curv -rusage /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/touch/rusage.mris_register.lh.dat ../surf/lh.sphere /home/stone/software/freesurfer_stable_600/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif ../surf/lh.sphere.reg 

#--------------------------------------------
#@# Surf Reg rh Wed May  8 21:44:55 CDT 2019

 mris_register -curv -rusage /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/touch/rusage.mris_register.rh.dat ../surf/rh.sphere /home/stone/software/freesurfer_stable_600/average/rh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif ../surf/rh.sphere.reg 

#--------------------------------------------
#@# Jacobian white lh Wed May  8 23:14:54 CDT 2019

 mris_jacobian ../surf/lh.white.preaparc ../surf/lh.sphere.reg ../surf/lh.jacobian_white 

#--------------------------------------------
#@# Jacobian white rh Wed May  8 23:14:58 CDT 2019

 mris_jacobian ../surf/rh.white.preaparc ../surf/rh.sphere.reg ../surf/rh.jacobian_white 

#--------------------------------------------
#@# AvgCurv lh Wed May  8 23:15:02 CDT 2019

 mrisp_paint -a 5 /home/stone/software/freesurfer_stable_600/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif#6 ../surf/lh.sphere.reg ../surf/lh.avg_curv 

#--------------------------------------------
#@# AvgCurv rh Wed May  8 23:15:07 CDT 2019

 mrisp_paint -a 5 /home/stone/software/freesurfer_stable_600/average/rh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif#6 ../surf/rh.sphere.reg ../surf/rh.avg_curv 

#-----------------------------------------
#@# Cortical Parc lh Wed May  8 23:15:11 CDT 2019

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 FS6EXPERT_subj01 lh ../surf/lh.sphere.reg /home/stone/software/freesurfer_stable_600/average/lh.DKaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/lh.aparc.annot 

#-----------------------------------------
#@# Cortical Parc rh Wed May  8 23:15:42 CDT 2019

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 FS6EXPERT_subj01 rh ../surf/rh.sphere.reg /home/stone/software/freesurfer_stable_600/average/rh.DKaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/rh.aparc.annot 

#--------------------------------------------
#@# Make Pial Surf lh Wed May  8 23:16:12 CDT 2019

 mris_make_surfaces -orig_white white.preaparc -orig_pial white.preaparc -aseg ../mri/aseg.presurf -mgz -T1 brain.finalsurfs FS6EXPERT_subj01 lh 

#--------------------------------------------
#@# Make Pial Surf rh Wed May  8 23:52:56 CDT 2019

 mris_make_surfaces -orig_white white.preaparc -orig_pial white.preaparc -aseg ../mri/aseg.presurf -mgz -T1 brain.finalsurfs FS6EXPERT_subj01 rh 

#--------------------------------------------
#@# Surf Volume lh Thu May  9 00:28:20 CDT 2019
#--------------------------------------------
#@# Surf Volume rh Thu May  9 00:28:30 CDT 2019
#--------------------------------------------
#@# Cortical ribbon mask Thu May  9 00:28:40 CDT 2019

 mris_volmask --aseg_name aseg.presurf --label_left_white 2 --label_left_ribbon 3 --label_right_white 41 --label_right_ribbon 42 --save_ribbon FS6EXPERT_subj01 

#-----------------------------------------
#@# Parcellation Stats lh Thu May  9 01:21:32 CDT 2019

 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.stats -b -a ../label/lh.aparc.annot -c ../label/aparc.annot.ctab FS6EXPERT_subj01 lh white 


 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.pial.stats -b -a ../label/lh.aparc.annot -c ../label/aparc.annot.ctab FS6EXPERT_subj01 lh pial 

#-----------------------------------------
#@# Parcellation Stats rh Thu May  9 01:25:12 CDT 2019

 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.stats -b -a ../label/rh.aparc.annot -c ../label/aparc.annot.ctab FS6EXPERT_subj01 rh white 


 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.pial.stats -b -a ../label/rh.aparc.annot -c ../label/aparc.annot.ctab FS6EXPERT_subj01 rh pial 

#-----------------------------------------
#@# Cortical Parc 2 lh Thu May  9 01:28:29 CDT 2019

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 FS6EXPERT_subj01 lh ../surf/lh.sphere.reg /home/stone/software/freesurfer_stable_600/average/lh.CDaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/lh.aparc.a2009s.annot 

#-----------------------------------------
#@# Cortical Parc 2 rh Thu May  9 01:29:11 CDT 2019

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 FS6EXPERT_subj01 rh ../surf/rh.sphere.reg /home/stone/software/freesurfer_stable_600/average/rh.CDaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/rh.aparc.a2009s.annot 

#-----------------------------------------
#@# Parcellation Stats 2 lh Thu May  9 01:29:46 CDT 2019

 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.a2009s.stats -b -a ../label/lh.aparc.a2009s.annot -c ../label/aparc.annot.a2009s.ctab FS6EXPERT_subj01 lh white 

#-----------------------------------------
#@# Parcellation Stats 2 rh Thu May  9 01:31:29 CDT 2019

 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.a2009s.stats -b -a ../label/rh.aparc.a2009s.annot -c ../label/aparc.annot.a2009s.ctab FS6EXPERT_subj01 rh white 

#-----------------------------------------
#@# Cortical Parc 3 lh Thu May  9 01:33:09 CDT 2019

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 FS6EXPERT_subj01 lh ../surf/lh.sphere.reg /home/stone/software/freesurfer_stable_600/average/lh.DKTaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/lh.aparc.DKTatlas.annot 

#-----------------------------------------
#@# Cortical Parc 3 rh Thu May  9 01:33:36 CDT 2019

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 FS6EXPERT_subj01 rh ../surf/rh.sphere.reg /home/stone/software/freesurfer_stable_600/average/rh.DKTaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/rh.aparc.DKTatlas.annot 

#-----------------------------------------
#@# Parcellation Stats 3 lh Thu May  9 01:34:03 CDT 2019

 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.DKTatlas.stats -b -a ../label/lh.aparc.DKTatlas.annot -c ../label/aparc.annot.DKTatlas.ctab FS6EXPERT_subj01 lh white 

#-----------------------------------------
#@# Parcellation Stats 3 rh Thu May  9 01:35:32 CDT 2019

 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.DKTatlas.stats -b -a ../label/rh.aparc.DKTatlas.annot -c ../label/aparc.annot.DKTatlas.ctab FS6EXPERT_subj01 rh white 

#-----------------------------------------
#@# WM/GM Contrast lh Thu May  9 01:37:06 CDT 2019

 pctsurfcon --s FS6EXPERT_subj01 --lh-only 

#-----------------------------------------
#@# WM/GM Contrast rh Thu May  9 01:37:20 CDT 2019

 pctsurfcon --s FS6EXPERT_subj01 --rh-only 

#-----------------------------------------
#@# Relabel Hypointensities Thu May  9 01:37:37 CDT 2019

 mri_relabel_hypointensities aseg.presurf.mgz ../surf aseg.presurf.hypos.mgz 

#-----------------------------------------
#@# AParc-to-ASeg aparc Thu May  9 01:38:41 CDT 2019

 mri_aparc2aseg --s FS6EXPERT_subj01 --volmask --aseg aseg.presurf.hypos --relabel mri/norm.mgz mri/transforms/talairach.m3z /home/stone/software/freesurfer_stable_600/average/RB_all_2016-05-10.vc700.gca mri/aseg.auto_noCCseg.label_intensities.txt 

#-----------------------------------------
#@# AParc-to-ASeg a2009s Thu May  9 01:48:16 CDT 2019

 mri_aparc2aseg --s FS6EXPERT_subj01 --volmask --aseg aseg.presurf.hypos --relabel mri/norm.mgz mri/transforms/talairach.m3z /home/stone/software/freesurfer_stable_600/average/RB_all_2016-05-10.vc700.gca mri/aseg.auto_noCCseg.label_intensities.txt --a2009s 

#-----------------------------------------
#@# AParc-to-ASeg DKTatlas Thu May  9 02:07:10 CDT 2019

 mri_aparc2aseg --s FS6EXPERT_subj01 --volmask --aseg aseg.presurf.hypos --relabel mri/norm.mgz mri/transforms/talairach.m3z /home/stone/software/freesurfer_stable_600/average/RB_all_2016-05-10.vc700.gca mri/aseg.auto_noCCseg.label_intensities.txt --annot aparc.DKTatlas --o mri/aparc.DKTatlas+aseg.mgz 

#-----------------------------------------
#@# APas-to-ASeg Thu May  9 02:24:54 CDT 2019

 apas2aseg --i aparc+aseg.mgz --o aseg.mgz 

#--------------------------------------------
#@# ASeg Stats Thu May  9 02:25:11 CDT 2019

 mri_segstats --seg mri/aseg.mgz --sum stats/aseg.stats --pv mri/norm.mgz --empty --brainmask mri/brainmask.mgz --brain-vol-from-seg --excludeid 0 --excl-ctxgmwm --supratent --subcortgray --in mri/norm.mgz --in-intensity-name norm --in-intensity-units MR --etiv --surf-wm-vol --surf-ctx-vol --totalgray --euler --ctab /home/stone/software/freesurfer_stable_600/ASegStatsLUT.txt --subject FS6EXPERT_subj01 

#-----------------------------------------
#@# WMParc Thu May  9 02:27:02 CDT 2019

 mri_aparc2aseg --s FS6EXPERT_subj01 --labelwm --hypo-as-wm --rip-unknown --volmask --o mri/wmparc.mgz --ctxseg aparc+aseg.mgz 


 mri_segstats --seg mri/wmparc.mgz --sum stats/wmparc.stats --pv mri/norm.mgz --excludeid 0 --brainmask mri/brainmask.mgz --in mri/norm.mgz --in-intensity-name norm --in-intensity-units MR --subject FS6EXPERT_subj01 --surf-wm-vol --ctab /home/stone/software/freesurfer_stable_600/WMParcStatsLUT.txt --etiv 

#--------------------------------------------
#@# BA_exvivo Labels lh Thu May  9 02:46:32 CDT 2019

 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA1_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.BA1_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA2_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.BA2_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA3a_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.BA3a_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA3b_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.BA3b_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA4a_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.BA4a_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA4p_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.BA4p_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA6_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.BA6_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA44_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.BA44_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA45_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.BA45_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.V1_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.V1_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.V2_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.V2_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.MT_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.MT_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.entorhinal_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.entorhinal_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.perirhinal_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.perirhinal_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA1_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.BA1_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA2_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.BA2_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA3a_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.BA3a_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA3b_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.BA3b_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA4a_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.BA4a_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA4p_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.BA4p_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA6_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.BA6_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA44_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.BA44_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA45_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.BA45_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.V1_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.V1_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.V2_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.V2_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.MT_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.MT_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.entorhinal_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.entorhinal_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.perirhinal_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./lh.perirhinal_exvivo.thresh.label --hemi lh --regmethod surface 


 mris_label2annot --s FS6EXPERT_subj01 --hemi lh --ctab /home/stone/software/freesurfer_stable_600/average/colortable_BA.txt --l lh.BA1_exvivo.label --l lh.BA2_exvivo.label --l lh.BA3a_exvivo.label --l lh.BA3b_exvivo.label --l lh.BA4a_exvivo.label --l lh.BA4p_exvivo.label --l lh.BA6_exvivo.label --l lh.BA44_exvivo.label --l lh.BA45_exvivo.label --l lh.V1_exvivo.label --l lh.V2_exvivo.label --l lh.MT_exvivo.label --l lh.entorhinal_exvivo.label --l lh.perirhinal_exvivo.label --a BA_exvivo --maxstatwinner --noverbose 


 mris_label2annot --s FS6EXPERT_subj01 --hemi lh --ctab /home/stone/software/freesurfer_stable_600/average/colortable_BA.txt --l lh.BA1_exvivo.thresh.label --l lh.BA2_exvivo.thresh.label --l lh.BA3a_exvivo.thresh.label --l lh.BA3b_exvivo.thresh.label --l lh.BA4a_exvivo.thresh.label --l lh.BA4p_exvivo.thresh.label --l lh.BA6_exvivo.thresh.label --l lh.BA44_exvivo.thresh.label --l lh.BA45_exvivo.thresh.label --l lh.V1_exvivo.thresh.label --l lh.V2_exvivo.thresh.label --l lh.MT_exvivo.thresh.label --l lh.entorhinal_exvivo.thresh.label --l lh.perirhinal_exvivo.thresh.label --a BA_exvivo.thresh --maxstatwinner --noverbose 


 mris_anatomical_stats -th3 -mgz -f ../stats/lh.BA_exvivo.stats -b -a ./lh.BA_exvivo.annot -c ./BA_exvivo.ctab FS6EXPERT_subj01 lh white 


 mris_anatomical_stats -th3 -mgz -f ../stats/lh.BA_exvivo.thresh.stats -b -a ./lh.BA_exvivo.thresh.annot -c ./BA_exvivo.thresh.ctab FS6EXPERT_subj01 lh white 

#--------------------------------------------
#@# BA_exvivo Labels rh Thu May  9 02:58:00 CDT 2019

 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA1_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.BA1_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA2_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.BA2_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA3a_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.BA3a_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA3b_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.BA3b_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA4a_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.BA4a_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA4p_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.BA4p_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA6_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.BA6_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA44_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.BA44_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA45_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.BA45_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.V1_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.V1_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.V2_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.V2_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.MT_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.MT_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.entorhinal_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.entorhinal_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.perirhinal_exvivo.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.perirhinal_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA1_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.BA1_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA2_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.BA2_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA3a_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.BA3a_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA3b_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.BA3b_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA4a_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.BA4a_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA4p_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.BA4p_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA6_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.BA6_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA44_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.BA44_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA45_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.BA45_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.V1_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.V1_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.V2_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.V2_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.MT_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.MT_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.entorhinal_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.entorhinal_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.perirhinal_exvivo.thresh.label --trgsubject FS6EXPERT_subj01 --trglabel ./rh.perirhinal_exvivo.thresh.label --hemi rh --regmethod surface 


 mris_label2annot --s FS6EXPERT_subj01 --hemi rh --ctab /home/stone/software/freesurfer_stable_600/average/colortable_BA.txt --l rh.BA1_exvivo.label --l rh.BA2_exvivo.label --l rh.BA3a_exvivo.label --l rh.BA3b_exvivo.label --l rh.BA4a_exvivo.label --l rh.BA4p_exvivo.label --l rh.BA6_exvivo.label --l rh.BA44_exvivo.label --l rh.BA45_exvivo.label --l rh.V1_exvivo.label --l rh.V2_exvivo.label --l rh.MT_exvivo.label --l rh.entorhinal_exvivo.label --l rh.perirhinal_exvivo.label --a BA_exvivo --maxstatwinner --noverbose 


 mris_label2annot --s FS6EXPERT_subj01 --hemi rh --ctab /home/stone/software/freesurfer_stable_600/average/colortable_BA.txt --l rh.BA1_exvivo.thresh.label --l rh.BA2_exvivo.thresh.label --l rh.BA3a_exvivo.thresh.label --l rh.BA3b_exvivo.thresh.label --l rh.BA4a_exvivo.thresh.label --l rh.BA4p_exvivo.thresh.label --l rh.BA6_exvivo.thresh.label --l rh.BA44_exvivo.thresh.label --l rh.BA45_exvivo.thresh.label --l rh.V1_exvivo.thresh.label --l rh.V2_exvivo.thresh.label --l rh.MT_exvivo.thresh.label --l rh.entorhinal_exvivo.thresh.label --l rh.perirhinal_exvivo.thresh.label --a BA_exvivo.thresh --maxstatwinner --noverbose 


 mris_anatomical_stats -th3 -mgz -f ../stats/rh.BA_exvivo.stats -b -a ./rh.BA_exvivo.annot -c ./BA_exvivo.ctab FS6EXPERT_subj01 rh white 


 mris_anatomical_stats -th3 -mgz -f ../stats/rh.BA_exvivo.thresh.stats -b -a ./rh.BA_exvivo.thresh.annot -c ./BA_exvivo.thresh.ctab FS6EXPERT_subj01 rh white 

#--------------------------------------------
#@# Segmentation of brainstem substructures  Thu May  9 03:08:31 CDT 2019

 /home/stone/software/freesurfer_stable_600/bin/segmentBS.sh /home/stone/software/freesurfer_stable_600/MCRv80 /home/stone/software/freesurfer_stable_600 FS6EXPERT_subj01 /home/stone-ext1/freesurfer/subjects 

See log file: /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/scripts/brainstem-structures.log
#--------------------------------------------
#@# Hippocampal Subfields processing (T1 + T2 volume) left Thu May  9 03:27:37 CDT 2019

 /home/stone/software/freesurfer_stable_600/bin/segmentSF_T1T2.sh /home/stone/software/freesurfer_stable_600/MCRv80 /home/stone/software/freesurfer_stable_600 FS6EXPERT_subj01 /home/stone-ext1/freesurfer/subjects /home/surly-raid1/kendrick-data/nsd/ppdata/NSD134-structurals/T2_0pt8_masked.nii.gz left HST1T2 

See log file: /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/scripts/hippocampal-subfields-T1T2.log
#--------------------------------------------
#@# Hippocampal Subfields processing (T1 + T2 volume) right Thu May  9 04:04:15 CDT 2019

 /home/stone/software/freesurfer_stable_600/bin/segmentSF_T1T2.sh /home/stone/software/freesurfer_stable_600/MCRv80 /home/stone/software/freesurfer_stable_600 FS6EXPERT_subj01 /home/stone-ext1/freesurfer/subjects /home/surly-raid1/kendrick-data/nsd/ppdata/NSD134-structurals/T2_0pt8_masked.nii.gz right HST1T2 

See log file: /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01/scripts/hippocampal-subfields-T1T2.log


#---------------------------------
# New invocation of recon-all Sat Jun 22 22:19:07 CDT 2019 
#--------------------------------------------
#@# Intensity Normalization2 Sat Jun 22 22:19:08 CDT 2019

 mri_normalize -f /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01_ver10/tmp/control.dat -mprage -noconform -aseg aseg.presurf.mgz -mask brainmask.mgz norm.mgz brain.mgz 

#--------------------------------------------
#@# Mask BFS Sat Jun 22 22:31:08 CDT 2019

 mri_mask -T 5 brain.mgz brainmask.mgz brain.finalsurfs.mgz 

#--------------------------------------------
#@# WM Segmentation Sat Jun 22 22:31:11 CDT 2019

 mri_binarize --i wm.mgz --min 255 --max 255 --o wm255.mgz --count wm255.txt 


 mri_binarize --i wm.mgz --min 1 --max 1 --o wm1.mgz --count wm1.txt 


 rm wm1.mgz wm255.mgz 


 cp wm.mgz wm.seg.mgz 


 mri_segment -keep -mprage brain.mgz wm.seg.mgz 


 mri_edit_wm_with_aseg -keep-in wm.seg.mgz brain.mgz aseg.presurf.mgz wm.asegedit.mgz 


 mri_pretess -keep wm.asegedit.mgz wm norm.mgz wm.mgz 

#--------------------------------------------
#@# Fill Sat Jun 22 22:36:42 CDT 2019

 mri_fill -a ../scripts/ponscc.cut.log -xform transforms/talairach.lta -segmentation aseg.auto_noCCseg.mgz wm.mgz filled.mgz 

#--------------------------------------------
#@# Tessellate lh Sat Jun 22 22:38:50 CDT 2019

 mri_pretess ../mri/filled.mgz 255 ../mri/norm.mgz ../mri/filled-pretess255.mgz 


 mri_tessellate ../mri/filled-pretess255.mgz 255 ../surf/lh.orig.nofix 


 rm -f ../mri/filled-pretess255.mgz 


 mris_extract_main_component ../surf/lh.orig.nofix ../surf/lh.orig.nofix 

#--------------------------------------------
#@# Tessellate rh Sat Jun 22 22:39:05 CDT 2019

 mri_pretess ../mri/filled.mgz 127 ../mri/norm.mgz ../mri/filled-pretess127.mgz 


 mri_tessellate ../mri/filled-pretess127.mgz 127 ../surf/rh.orig.nofix 


 rm -f ../mri/filled-pretess127.mgz 


 mris_extract_main_component ../surf/rh.orig.nofix ../surf/rh.orig.nofix 

#--------------------------------------------
#@# Smooth1 lh Sat Jun 22 22:39:22 CDT 2019

 mris_smooth -nw -seed 1234 ../surf/lh.orig.nofix ../surf/lh.smoothwm.nofix 

#--------------------------------------------
#@# Smooth1 rh Sat Jun 22 22:39:36 CDT 2019

 mris_smooth -nw -seed 1234 ../surf/rh.orig.nofix ../surf/rh.smoothwm.nofix 

#--------------------------------------------
#@# Inflation1 lh Sat Jun 22 22:39:50 CDT 2019

 mris_inflate -no-save-sulc -n 50 ../surf/lh.smoothwm.nofix ../surf/lh.inflated.nofix 

#--------------------------------------------
#@# Inflation1 rh Sat Jun 22 22:43:43 CDT 2019

 mris_inflate -no-save-sulc -n 50 ../surf/rh.smoothwm.nofix ../surf/rh.inflated.nofix 

#--------------------------------------------
#@# QSphere lh Sat Jun 22 22:47:16 CDT 2019

 mris_sphere -q -seed 1234 ../surf/lh.inflated.nofix ../surf/lh.qsphere.nofix 

#--------------------------------------------
#@# QSphere rh Sat Jun 22 22:51:33 CDT 2019

 mris_sphere -q -seed 1234 ../surf/rh.inflated.nofix ../surf/rh.qsphere.nofix 

#--------------------------------------------
#@# Fix Topology Copy lh Sat Jun 22 22:55:16 CDT 2019

 cp ../surf/lh.orig.nofix ../surf/lh.orig 


 cp ../surf/lh.inflated.nofix ../surf/lh.inflated 

#--------------------------------------------
#@# Fix Topology Copy rh Sat Jun 22 22:55:16 CDT 2019

 cp ../surf/rh.orig.nofix ../surf/rh.orig 


 cp ../surf/rh.inflated.nofix ../surf/rh.inflated 

#@# Fix Topology lh Sat Jun 22 22:55:16 CDT 2019

 mris_fix_topology -rusage /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01_ver10/touch/rusage.mris_fix_topology.lh.dat -mgz -sphere qsphere.nofix -ga -seed 1234 FS6EXPERT_subj01_ver10 lh 

#@# Fix Topology rh Sat Jun 22 23:22:01 CDT 2019

 mris_fix_topology -rusage /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01_ver10/touch/rusage.mris_fix_topology.rh.dat -mgz -sphere qsphere.nofix -ga -seed 1234 FS6EXPERT_subj01_ver10 rh 


 mris_euler_number ../surf/lh.orig 


 mris_euler_number ../surf/rh.orig 


 mris_remove_intersection ../surf/lh.orig ../surf/lh.orig 


 rm ../surf/lh.inflated 


 mris_remove_intersection ../surf/rh.orig ../surf/rh.orig 


 rm ../surf/rh.inflated 

#--------------------------------------------
#@# Make White Surf lh Sun Jun 23 00:05:01 CDT 2019

 mris_make_surfaces -aseg ../mri/aseg.presurf -white white.preaparc -noaparc -whiteonly -mgz -T1 brain.finalsurfs FS6EXPERT_subj01_ver10 lh 

#--------------------------------------------
#@# Make White Surf rh Sun Jun 23 00:11:36 CDT 2019

 mris_make_surfaces -aseg ../mri/aseg.presurf -white white.preaparc -noaparc -whiteonly -mgz -T1 brain.finalsurfs FS6EXPERT_subj01_ver10 rh 

#--------------------------------------------
#@# Smooth2 lh Sun Jun 23 00:18:07 CDT 2019

 mris_smooth -n 3 -nw -seed 1234 ../surf/lh.white.preaparc ../surf/lh.smoothwm 

#--------------------------------------------
#@# Smooth2 rh Sun Jun 23 00:18:21 CDT 2019

 mris_smooth -n 3 -nw -seed 1234 ../surf/rh.white.preaparc ../surf/rh.smoothwm 

#--------------------------------------------
#@# Inflation2 lh Sun Jun 23 00:18:36 CDT 2019

 mris_inflate -rusage /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01_ver10/touch/rusage.mris_inflate.lh.dat -n 50 ../surf/lh.smoothwm ../surf/lh.inflated 

#--------------------------------------------
#@# Inflation2 rh Sun Jun 23 00:20:33 CDT 2019

 mris_inflate -rusage /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01_ver10/touch/rusage.mris_inflate.rh.dat -n 50 ../surf/rh.smoothwm ../surf/rh.inflated 

#--------------------------------------------
#@# Curv .H and .K lh Sun Jun 23 00:23:03 CDT 2019

 mris_curvature -w lh.white.preaparc 


 mris_curvature -thresh .999 -n -a 5 -w -distances 10 10 lh.inflated 

#--------------------------------------------
#@# Curv .H and .K rh Sun Jun 23 00:25:39 CDT 2019

 mris_curvature -w rh.white.preaparc 


 mris_curvature -thresh .999 -n -a 5 -w -distances 10 10 rh.inflated 


#-----------------------------------------
#@# Curvature Stats lh Sun Jun 23 00:28:23 CDT 2019

 mris_curvature_stats -m --writeCurvatureFiles -G -o ../stats/lh.curv.stats -F smoothwm FS6EXPERT_subj01_ver10 lh curv sulc 


#-----------------------------------------
#@# Curvature Stats rh Sun Jun 23 00:28:34 CDT 2019

 mris_curvature_stats -m --writeCurvatureFiles -G -o ../stats/rh.curv.stats -F smoothwm FS6EXPERT_subj01_ver10 rh curv sulc 

#--------------------------------------------
#@# Sphere lh Sun Jun 23 00:28:43 CDT 2019

 mris_sphere -rusage /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01_ver10/touch/rusage.mris_sphere.lh.dat -seed 1234 ../surf/lh.inflated ../surf/lh.sphere 

#--------------------------------------------
#@# Sphere rh Sun Jun 23 00:49:44 CDT 2019

 mris_sphere -rusage /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01_ver10/touch/rusage.mris_sphere.rh.dat -seed 1234 ../surf/rh.inflated ../surf/rh.sphere 

#--------------------------------------------
#@# Surf Reg lh Sun Jun 23 01:13:29 CDT 2019

 mris_register -curv -rusage /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01_ver10/touch/rusage.mris_register.lh.dat ../surf/lh.sphere /home/stone/software/freesurfer_stable_600/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif ../surf/lh.sphere.reg 

#--------------------------------------------
#@# Surf Reg rh Sun Jun 23 02:36:34 CDT 2019

 mris_register -curv -rusage /home/stone-ext1/freesurfer/subjects/FS6EXPERT_subj01_ver10/touch/rusage.mris_register.rh.dat ../surf/rh.sphere /home/stone/software/freesurfer_stable_600/average/rh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif ../surf/rh.sphere.reg 

#--------------------------------------------
#@# Jacobian white lh Sun Jun 23 04:05:05 CDT 2019

 mris_jacobian ../surf/lh.white.preaparc ../surf/lh.sphere.reg ../surf/lh.jacobian_white 

#--------------------------------------------
#@# Jacobian white rh Sun Jun 23 04:05:09 CDT 2019

 mris_jacobian ../surf/rh.white.preaparc ../surf/rh.sphere.reg ../surf/rh.jacobian_white 

#--------------------------------------------
#@# AvgCurv lh Sun Jun 23 04:05:13 CDT 2019

 mrisp_paint -a 5 /home/stone/software/freesurfer_stable_600/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif#6 ../surf/lh.sphere.reg ../surf/lh.avg_curv 

#--------------------------------------------
#@# AvgCurv rh Sun Jun 23 04:05:17 CDT 2019

 mrisp_paint -a 5 /home/stone/software/freesurfer_stable_600/average/rh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif#6 ../surf/rh.sphere.reg ../surf/rh.avg_curv 

#-----------------------------------------
#@# Cortical Parc lh Sun Jun 23 04:05:21 CDT 2019

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 FS6EXPERT_subj01_ver10 lh ../surf/lh.sphere.reg /home/stone/software/freesurfer_stable_600/average/lh.DKaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/lh.aparc.annot 

#-----------------------------------------
#@# Cortical Parc rh Sun Jun 23 04:05:54 CDT 2019

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 FS6EXPERT_subj01_ver10 rh ../surf/rh.sphere.reg /home/stone/software/freesurfer_stable_600/average/rh.DKaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/rh.aparc.annot 

#--------------------------------------------
#@# Make Pial Surf lh Sun Jun 23 04:06:26 CDT 2019

 mris_make_surfaces -orig_white white.preaparc -orig_pial white.preaparc -aseg ../mri/aseg.presurf -mgz -T1 brain.finalsurfs FS6EXPERT_subj01_ver10 lh 

#--------------------------------------------
#@# Make Pial Surf rh Sun Jun 23 04:40:07 CDT 2019

 mris_make_surfaces -orig_white white.preaparc -orig_pial white.preaparc -aseg ../mri/aseg.presurf -mgz -T1 brain.finalsurfs FS6EXPERT_subj01_ver10 rh 

#--------------------------------------------
#@# Surf Volume lh Sun Jun 23 05:03:55 CDT 2019
#--------------------------------------------
#@# Surf Volume rh Sun Jun 23 05:04:00 CDT 2019
#--------------------------------------------
#@# Cortical ribbon mask Sun Jun 23 05:04:06 CDT 2019

 mris_volmask --aseg_name aseg.presurf --label_left_white 2 --label_left_ribbon 3 --label_right_white 41 --label_right_ribbon 42 --save_ribbon FS6EXPERT_subj01_ver10 

#-----------------------------------------
#@# Parcellation Stats lh Sun Jun 23 05:29:21 CDT 2019

 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.stats -b -a ../label/lh.aparc.annot -c ../label/aparc.annot.ctab FS6EXPERT_subj01_ver10 lh white 


 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.pial.stats -b -a ../label/lh.aparc.annot -c ../label/aparc.annot.ctab FS6EXPERT_subj01_ver10 lh pial 

#-----------------------------------------
#@# Parcellation Stats rh Sun Jun 23 05:31:31 CDT 2019

 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.stats -b -a ../label/rh.aparc.annot -c ../label/aparc.annot.ctab FS6EXPERT_subj01_ver10 rh white 


 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.pial.stats -b -a ../label/rh.aparc.annot -c ../label/aparc.annot.ctab FS6EXPERT_subj01_ver10 rh pial 

#-----------------------------------------
#@# Cortical Parc 2 lh Sun Jun 23 05:33:42 CDT 2019

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 FS6EXPERT_subj01_ver10 lh ../surf/lh.sphere.reg /home/stone/software/freesurfer_stable_600/average/lh.CDaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/lh.aparc.a2009s.annot 

#-----------------------------------------
#@# Cortical Parc 2 rh Sun Jun 23 05:34:06 CDT 2019

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 FS6EXPERT_subj01_ver10 rh ../surf/rh.sphere.reg /home/stone/software/freesurfer_stable_600/average/rh.CDaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/rh.aparc.a2009s.annot 

#-----------------------------------------
#@# Parcellation Stats 2 lh Sun Jun 23 05:34:32 CDT 2019

 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.a2009s.stats -b -a ../label/lh.aparc.a2009s.annot -c ../label/aparc.annot.a2009s.ctab FS6EXPERT_subj01_ver10 lh white 

#-----------------------------------------
#@# Parcellation Stats 2 rh Sun Jun 23 05:35:37 CDT 2019

 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.a2009s.stats -b -a ../label/rh.aparc.a2009s.annot -c ../label/aparc.annot.a2009s.ctab FS6EXPERT_subj01_ver10 rh white 

#-----------------------------------------
#@# Cortical Parc 3 lh Sun Jun 23 05:36:46 CDT 2019

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 FS6EXPERT_subj01_ver10 lh ../surf/lh.sphere.reg /home/stone/software/freesurfer_stable_600/average/lh.DKTaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/lh.aparc.DKTatlas.annot 

#-----------------------------------------
#@# Cortical Parc 3 rh Sun Jun 23 05:37:06 CDT 2019

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 FS6EXPERT_subj01_ver10 rh ../surf/rh.sphere.reg /home/stone/software/freesurfer_stable_600/average/rh.DKTaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/rh.aparc.DKTatlas.annot 

#-----------------------------------------
#@# Parcellation Stats 3 lh Sun Jun 23 05:37:29 CDT 2019

 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.DKTatlas.stats -b -a ../label/lh.aparc.DKTatlas.annot -c ../label/aparc.annot.DKTatlas.ctab FS6EXPERT_subj01_ver10 lh white 

#-----------------------------------------
#@# Parcellation Stats 3 rh Sun Jun 23 05:38:37 CDT 2019

 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.DKTatlas.stats -b -a ../label/rh.aparc.DKTatlas.annot -c ../label/aparc.annot.DKTatlas.ctab FS6EXPERT_subj01_ver10 rh white 

#-----------------------------------------
#@# WM/GM Contrast lh Sun Jun 23 05:39:44 CDT 2019

 pctsurfcon --s FS6EXPERT_subj01_ver10 --lh-only 

#-----------------------------------------
#@# WM/GM Contrast rh Sun Jun 23 05:39:54 CDT 2019

 pctsurfcon --s FS6EXPERT_subj01_ver10 --rh-only 

#-----------------------------------------
#@# Relabel Hypointensities Sun Jun 23 05:40:05 CDT 2019

 mri_relabel_hypointensities aseg.presurf.mgz ../surf aseg.presurf.hypos.mgz 

#-----------------------------------------
#@# AParc-to-ASeg aparc Sun Jun 23 05:40:50 CDT 2019

 mri_aparc2aseg --s FS6EXPERT_subj01_ver10 --volmask --aseg aseg.presurf.hypos --relabel mri/norm.mgz mri/transforms/talairach.m3z /home/stone/software/freesurfer_stable_600/average/RB_all_2016-05-10.vc700.gca mri/aseg.auto_noCCseg.label_intensities.txt 

#-----------------------------------------
#@# AParc-to-ASeg a2009s Sun Jun 23 05:52:06 CDT 2019

 mri_aparc2aseg --s FS6EXPERT_subj01_ver10 --volmask --aseg aseg.presurf.hypos --relabel mri/norm.mgz mri/transforms/talairach.m3z /home/stone/software/freesurfer_stable_600/average/RB_all_2016-05-10.vc700.gca mri/aseg.auto_noCCseg.label_intensities.txt --a2009s 

#-----------------------------------------
#@# AParc-to-ASeg DKTatlas Sun Jun 23 06:03:07 CDT 2019

 mri_aparc2aseg --s FS6EXPERT_subj01_ver10 --volmask --aseg aseg.presurf.hypos --relabel mri/norm.mgz mri/transforms/talairach.m3z /home/stone/software/freesurfer_stable_600/average/RB_all_2016-05-10.vc700.gca mri/aseg.auto_noCCseg.label_intensities.txt --annot aparc.DKTatlas --o mri/aparc.DKTatlas+aseg.mgz 

#-----------------------------------------
#@# APas-to-ASeg Sun Jun 23 06:14:06 CDT 2019

 apas2aseg --i aparc+aseg.mgz --o aseg.mgz 

#--------------------------------------------
#@# ASeg Stats Sun Jun 23 06:14:17 CDT 2019

 mri_segstats --seg mri/aseg.mgz --sum stats/aseg.stats --pv mri/norm.mgz --empty --brainmask mri/brainmask.mgz --brain-vol-from-seg --excludeid 0 --excl-ctxgmwm --supratent --subcortgray --in mri/norm.mgz --in-intensity-name norm --in-intensity-units MR --etiv --surf-wm-vol --surf-ctx-vol --totalgray --euler --ctab /home/stone/software/freesurfer_stable_600/ASegStatsLUT.txt --subject FS6EXPERT_subj01_ver10 

#-----------------------------------------
#@# WMParc Sun Jun 23 06:15:23 CDT 2019

 mri_aparc2aseg --s FS6EXPERT_subj01_ver10 --labelwm --hypo-as-wm --rip-unknown --volmask --o mri/wmparc.mgz --ctxseg aparc+aseg.mgz 


 mri_segstats --seg mri/wmparc.mgz --sum stats/wmparc.stats --pv mri/norm.mgz --excludeid 0 --brainmask mri/brainmask.mgz --in mri/norm.mgz --in-intensity-name norm --in-intensity-units MR --subject FS6EXPERT_subj01_ver10 --surf-wm-vol --ctab /home/stone/software/freesurfer_stable_600/WMParcStatsLUT.txt --etiv 

#--------------------------------------------
#@# BA_exvivo Labels lh Sun Jun 23 06:25:11 CDT 2019

 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA1_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.BA1_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA2_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.BA2_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA3a_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.BA3a_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA3b_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.BA3b_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA4a_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.BA4a_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA4p_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.BA4p_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA6_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.BA6_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA44_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.BA44_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA45_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.BA45_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.V1_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.V1_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.V2_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.V2_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.MT_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.MT_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.entorhinal_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.entorhinal_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.perirhinal_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.perirhinal_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA1_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.BA1_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA2_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.BA2_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA3a_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.BA3a_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA3b_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.BA3b_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA4a_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.BA4a_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA4p_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.BA4p_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA6_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.BA6_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA44_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.BA44_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.BA45_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.BA45_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.V1_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.V1_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.V2_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.V2_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.MT_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.MT_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.entorhinal_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.entorhinal_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/lh.perirhinal_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./lh.perirhinal_exvivo.thresh.label --hemi lh --regmethod surface 


 mris_label2annot --s FS6EXPERT_subj01_ver10 --hemi lh --ctab /home/stone/software/freesurfer_stable_600/average/colortable_BA.txt --l lh.BA1_exvivo.label --l lh.BA2_exvivo.label --l lh.BA3a_exvivo.label --l lh.BA3b_exvivo.label --l lh.BA4a_exvivo.label --l lh.BA4p_exvivo.label --l lh.BA6_exvivo.label --l lh.BA44_exvivo.label --l lh.BA45_exvivo.label --l lh.V1_exvivo.label --l lh.V2_exvivo.label --l lh.MT_exvivo.label --l lh.entorhinal_exvivo.label --l lh.perirhinal_exvivo.label --a BA_exvivo --maxstatwinner --noverbose 


 mris_label2annot --s FS6EXPERT_subj01_ver10 --hemi lh --ctab /home/stone/software/freesurfer_stable_600/average/colortable_BA.txt --l lh.BA1_exvivo.thresh.label --l lh.BA2_exvivo.thresh.label --l lh.BA3a_exvivo.thresh.label --l lh.BA3b_exvivo.thresh.label --l lh.BA4a_exvivo.thresh.label --l lh.BA4p_exvivo.thresh.label --l lh.BA6_exvivo.thresh.label --l lh.BA44_exvivo.thresh.label --l lh.BA45_exvivo.thresh.label --l lh.V1_exvivo.thresh.label --l lh.V2_exvivo.thresh.label --l lh.MT_exvivo.thresh.label --l lh.entorhinal_exvivo.thresh.label --l lh.perirhinal_exvivo.thresh.label --a BA_exvivo.thresh --maxstatwinner --noverbose 


 mris_anatomical_stats -th3 -mgz -f ../stats/lh.BA_exvivo.stats -b -a ./lh.BA_exvivo.annot -c ./BA_exvivo.ctab FS6EXPERT_subj01_ver10 lh white 


 mris_anatomical_stats -th3 -mgz -f ../stats/lh.BA_exvivo.thresh.stats -b -a ./lh.BA_exvivo.thresh.annot -c ./BA_exvivo.thresh.ctab FS6EXPERT_subj01_ver10 lh white 

#--------------------------------------------
#@# BA_exvivo Labels rh Sun Jun 23 06:32:32 CDT 2019

 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA1_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.BA1_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA2_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.BA2_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA3a_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.BA3a_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA3b_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.BA3b_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA4a_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.BA4a_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA4p_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.BA4p_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA6_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.BA6_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA44_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.BA44_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA45_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.BA45_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.V1_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.V1_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.V2_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.V2_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.MT_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.MT_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.entorhinal_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.entorhinal_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.perirhinal_exvivo.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.perirhinal_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA1_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.BA1_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA2_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.BA2_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA3a_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.BA3a_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA3b_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.BA3b_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA4a_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.BA4a_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA4p_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.BA4p_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA6_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.BA6_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA44_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.BA44_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.BA45_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.BA45_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.V1_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.V1_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.V2_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.V2_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.MT_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.MT_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.entorhinal_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.entorhinal_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/stone-ext1/freesurfer/subjects/fsaverage/label/rh.perirhinal_exvivo.thresh.label --trgsubject FS6EXPERT_subj01_ver10 --trglabel ./rh.perirhinal_exvivo.thresh.label --hemi rh --regmethod surface 


 mris_label2annot --s FS6EXPERT_subj01_ver10 --hemi rh --ctab /home/stone/software/freesurfer_stable_600/average/colortable_BA.txt --l rh.BA1_exvivo.label --l rh.BA2_exvivo.label --l rh.BA3a_exvivo.label --l rh.BA3b_exvivo.label --l rh.BA4a_exvivo.label --l rh.BA4p_exvivo.label --l rh.BA6_exvivo.label --l rh.BA44_exvivo.label --l rh.BA45_exvivo.label --l rh.V1_exvivo.label --l rh.V2_exvivo.label --l rh.MT_exvivo.label --l rh.entorhinal_exvivo.label --l rh.perirhinal_exvivo.label --a BA_exvivo --maxstatwinner --noverbose 


 mris_label2annot --s FS6EXPERT_subj01_ver10 --hemi rh --ctab /home/stone/software/freesurfer_stable_600/average/colortable_BA.txt --l rh.BA1_exvivo.thresh.label --l rh.BA2_exvivo.thresh.label --l rh.BA3a_exvivo.thresh.label --l rh.BA3b_exvivo.thresh.label --l rh.BA4a_exvivo.thresh.label --l rh.BA4p_exvivo.thresh.label --l rh.BA6_exvivo.thresh.label --l rh.BA44_exvivo.thresh.label --l rh.BA45_exvivo.thresh.label --l rh.V1_exvivo.thresh.label --l rh.V2_exvivo.thresh.label --l rh.MT_exvivo.thresh.label --l rh.entorhinal_exvivo.thresh.label --l rh.perirhinal_exvivo.thresh.label --a BA_exvivo.thresh --maxstatwinner --noverbose 


 mris_anatomical_stats -th3 -mgz -f ../stats/rh.BA_exvivo.stats -b -a ./rh.BA_exvivo.annot -c ./BA_exvivo.ctab FS6EXPERT_subj01_ver10 rh white 


 mris_anatomical_stats -th3 -mgz -f ../stats/rh.BA_exvivo.thresh.stats -b -a ./rh.BA_exvivo.thresh.annot -c ./BA_exvivo.thresh.ctab FS6EXPERT_subj01_ver10 rh white 

