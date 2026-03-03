一、训练配置
1、所有配置文件位于configs里面：
configs/deimv2:  主配置文件夹。
配置的yaml文件包括：           dataset       数据集路径及类别参数；
runtime       一些重要超参数的配置
optimizer     优化器类型选择及配置
deimv2                 模型基础结构设置
2、训练步骤：使用coco格式数据集在dataset里配置路径等
选择合适的模型大小、输入分辨率、训练轮次等
                                 

二、其他说明
1、deimv2目前仅支持分类和目标检测任务
2、有两种骨架：hgnet、dinov3对应cnn和vit架构
3、可以先改动runtime里面的参数，对模型训练影响很大。
