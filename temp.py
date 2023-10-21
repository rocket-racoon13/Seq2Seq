class Sample:
    def __init__(self):
        self.a = 10
        self.b = 11
        self.c = 12
        
    def forward(self):
        for param in [self.a, self.b, self.c]:
            param -= 1
        
        print(self.a, self.b, self.c)
        
sample = Sample()
sample.forward()