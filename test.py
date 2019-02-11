import time

class test:
    def test(self):
        pass
"""
4.105237722396851
4.414266586303711

3.648768186569214
3.707662582397461

3.6752700805664062
4.047108888626099
"""
starttime = time.time()
batch_time = 0
for i in range(1000000):
    # time.sleep(1)
    try:
        batch_time = time.time()
    except KeyboardInterrupt:
        break
print(time.time() - starttime)


starttime = time.time()
batch_time = 0
for i in range(1000000):
    # time.sleep(1)
    if 1:
        pass
    if 1:
        pass
    try:
        batch_time = time.time()
    except KeyboardInterrupt:
        break
print(time.time() - starttime)

starttime = time.time()
batch_time = 0
x = test()
for i in range(1000000):
    # time.sleep(1)
    x.test()
    x.test()
    try:
        batch_time = time.time()
    except KeyboardInterrupt:
        break
print(time.time() - starttime)
