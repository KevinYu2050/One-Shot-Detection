

#env \
python=3.5.2\
torch=1.2.0\
cuda=10.0\
cudnn=7.5.0

#train-test\
python train_online.py\


#手动标注分割label
cd script/roipoly.py/ \
python get_mask.py

#script下脚本功能\
get_bbox_mask.py：将原始标注的json文件获得合并；生成用于训练的二值化分割图\
rotate_img.py:旋转图片\
paste.py:将旋转得到的图片贴到背景数据上用于数据增强\
mask2bbox.py:从预测mask获得bbox\
get_PR.py:简化版计算precision-recall


#pipeline\
stpe1:执行train_onlien.py生成mask图片\
step2:执行mask2bbox.py,从mask图片生产det_17.json\
step3: 利用get_bbox_mask.py 生成gt_dict.json \
step4:: 利用get_every_gt.py生成每一类的gt_17.json\
step5:执行get_PR获得PR
