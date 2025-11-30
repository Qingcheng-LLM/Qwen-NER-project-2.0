
#*args用于处理不确定数量的位置参数，结构是元组。*kwargs用于处理不确定数量的关键字参数，结构是字典。
class my_Config:
    #解析完字典的k,v放入结构配置类实例
    def __init__(self,**kwargs):
        for k, v in kwargs.items():    #遍历**kwargs（键值对参数）**kwargs是Python语言的内置特性。**​​表示专门用于收集所有关键字参数，将参数收集到字典(dict)中
            setattr(self, k, v)        #使用setattr动态设置属性值，键是参数名，值是参数值
    #更新参数
    def merge_attr(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    #返回配置的字符串表示形式，用于打印配置信息。
    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])
