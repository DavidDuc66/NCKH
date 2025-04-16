#import epyt
from epyt import epanet
import numpy as np
d = epanet('FILEGOCCUADUY.inp') 

np.set_printoptions(precision=40)

Delta = np.empty((0, 0))
KQ = np.empty((0, 0))
KBN = np.empty((0, 0))
KQRR = np.empty((0, 0))
KR = np.empty((0, 0))
M1 = np.empty((0, 0))
IM = np.empty((0, 0))
K_L = np.empty((0, 0))
SST = np.empty((0, 0))
######## Thêm pattern mới vào file inp
TG = len(d.Pattern[0]) 
Patternid = 'RRR'
import numpy as np
patternMult = np.ones(TG)
PatternIndex = d.addPattern(Patternid, patternMult )

########### Chạy thủy lực lúc chưa gán rò rỉ
hyd_res_2 =d.getComputedTimeSeries()

######## Xuất kết quả áp suất giờ thấp điểm
J=d.getNodeJunctionIndex()
J= np.array(J)
J = J.reshape(-1, 1)
F= hyd_res_2.Demand[:,:len(J)]#Flow
F=np.array(F.T)
column_sums = np.sum(F, axis=0) # sum các cột
GTD = np.argmin(column_sums) #dùng để xd cột có tổng thất nhất
KB0 = hyd_res_2.Pressure[GTD,:len(J)]
KB0=KB0.T # chuyển hàng thành cột
KB0=np.array(KB0)# chuyển sang np
KB0 = KB0.reshape(-1, 1)
KQ = np.empty((KB0.shape[0], 0))
KQRR = np.empty((KB0.shape[0], 0))
Delta = np.empty((KB0.shape[0], 0))
KR = np.empty((KB0.shape[0], 0))

####3#thuat toan cho tung nut
for i in J:
    LR = [5, 10, 20, 40, 60]
    for LL in LR: 
        d.addNodeJunctionDemand( i, LL,'RR','Ro Ri')
        hyd_res_2 = d.getComputedTimeSeries()
        KB1 =hyd_res_2.Pressure[GTD,:len(J)];
        KB1=np.array(KB1)
        KB1 = KB1.reshape(-1, 1)
        KR = np.concatenate((KR, KB1), axis=1); #KR la ap suat voi kich ban ro ri
        KBN = KB1 - KB0  #KBN chenh lech ap suat voi kich ban ko ro ri
        Delta = np.concatenate((Delta, KBN), axis=1)
        d.addNodeJunctionDemand(i,-LL, 'RR', 'Ro Ri');
    KQ = np.concatenate((KQ, Delta), axis=1); #KQ la tong hop chenh lech ap suat tai mot nut
    KQRR=np.concatenate((KQRR, KR), axis=1); #KQRR la tong hop ap suat tai mot nut voi kich ban ro ri
    Delta = np.empty((KB0.shape[0], 0))
    KR = np.empty((KB0.shape[0], 0))
#làm tròn số
HSLT=-1000 
KQ=KQ*HSLT
KQ= np.round(KQ, decimals=0)

#set tool và def entropy
from collections import Counter
def shannon_entropy(data):
    # Calculate the probabilities
    counts = Counter(data)
    total_count = len(data)
    probabilities = np.array([count / total_count for count in counts.values()])
    
    # Calculate the entropy
    return -np.sum(probabilities * np.log2(probabilities))

# tao M5 (lưu lượng rò rỉ 5l/s)
M5 = np.empty((KB0.shape[0], 0))
for i in J:
    M5=np.concatenate((M5, KQ[:, i*5-5]), axis=1)

############################################################################################################################
#tínhentropylúc đầu 
NTT= d.getNodeJunctionDemandIndex()
sizeMau = np.shape(NTT)
SN = sizeMau[1]
nj=SN
mau = np.arange(1, nj + 1)
mau = mau.reshape(-1, 1)
entr = np.empty((0, 1))
A = np.empty((0, 1))
for i in range(1, nj+1):
    row = M5[i-1, :]
    A = shannon_entropy(row)
    A= np.array([A])
    A = A.reshape(-1, 1) 
    entr=np.concatenate((entr, A), axis=0)
    A = np.empty((0, 1))


#Xuất giá trị có Entropy lớn nhất
ANS = np.empty((1, 0))
ANS = np.argmax(entr) 
row, col = np.unravel_index(ANS, entr.shape)
#print(f"Phần tử lớn nhất là {entr[row, col]} nằm tại vị trí: hàng {row}, cột {col}")
SS= np.empty((1, 0))
row=row.reshape(-1, 1) 
SS= np.concatenate((SS, row), axis=1);
SS1=SS.astype(int).flatten()  #bắt buộc mảng dùng để trích xuất mảng khác thuộc type int. Kiểm tra = lệnh "print(A.dtype)""

#xuất các vị trí bị lặp
A=M5[SS1,:] 
A=A.T
# Tìm các giá trị duy nhất và vị trí của chúng (giữ nguyên thứ tự xuất hiện - stable)
_, w = np.unique(A, return_index=True)
# Sắp xếp chỉ số w để giữ thứ tự xuất hiện ban đầu (stable)
w = np.sort(w)
# Xác định các vị trí của phần tử bị lặp lại
all_indices = np.arange(len(A))  # Index bắt đầu từ 0
dup = np.setdiff1d(all_indices, w)  # Không cần +1 vì index bắt đầu từ 0
dup=dup.reshape(-1, 1) 

#tính độ chính xác
sizeMau = dup.shape
sz = sizeMau[0] 
dcx = (nj - sz + 1) / nj
dcx = np.array([dcx])
dcx=dcx.reshape(-1, 1) 
dochinhxac = np.empty((1, 0))
dochinhxac = np.concatenate((dochinhxac, dcx), axis=1);

#####################################################################################################################
#####vòng lặp cho thuật toán 
SST=SS+1
mau=dup
giatribilap=np.empty((0, 0))
entr = np.empty((0, 1))
A = np.empty((0, 1))
for i in range(1, nj+1):
     if i not in SST:
         row = M5[i-1, mau.flatten()]
         A = shannon_entropy(row)
     else:
         A=0
     A= np.array([A])
     A = A.reshape(-1, 1) 
     entr=np.concatenate((entr, A), axis=0)
     A = np.empty((0, 1))    

#tinhMIcuatatcacacnutsovoinutduocchon(tainhugvitribilap)
#thuật toán MIimport numpy as np
from sklearn.feature_selection import mutual_info_classif

def calculate_featurewise_mutual_information(X, y):
    # X is a 2D array (samples x features)
    # y is a 1D array (labels or target variable)
    return mutual_info_classif(X, y, discrete_features=True)

#tạo matrix chứa các giá trị bị lặp của các sensor được chọn 
matrix=np.empty((mau.shape[0], 0))
MTT=np.empty((mau.shape[0], 0))
for i in range(1, nj+1):
    if i not in SST:
        continue
    else:  
        MTT=M5[i-1, mau.flatten()]
        MTT=MTT.reshape(-1, 1) 
    matrix=np.concatenate((matrix, MTT), axis=1)

#mã hóa Matrix
_, matrix_MH = np.unique(matrix, axis=0, return_inverse=True) 
matrix_MH = matrix_MH + 1  # Để bắt đầu từ 1 thay vì 0
matrix_MH=matrix_MH.reshape(-1, 1)
#tính MI 
array=np.empty((0, 1))
MI = np.empty((0, 1))
MIT = np.empty((0, 1))
for i in range(1, nj+1):
    array=M5[i-1, mau.flatten()]
    if i not in SST:
        MIT = calculate_featurewise_mutual_information(matrix_MH, array)
    else: 
        MIT=0
    MIT=MIT
    MIT= np.array([MIT])
    MIT= MIT.reshape(-1, 1)
    MI=np.concatenate((MI, MIT), axis=0)

#tinhentropychiaMI
result = np.empty((0, 1))
for i in range(0, nj):
    if i not in SST:
        T1 = entr[i, 0]
        T2 = MI[i, 0]
        T = T1/T2
        if T2 != 0:
            T = T1 / T2
        else:
            T = 0
    else:
        T=0
    T= np.array([T])
    T= T.reshape(-1, 1)
    result=np.concatenate((result, T), axis=0)

#lựa chọn nút có kết quả lớn nhất và gán vào nút được chọn
max_index = np.argmax(result)
max_index=max_index.reshape(-1, 1)
SS= np.concatenate((SS, max_index), axis=1);
SS1=max_index.astype(int).flatten()

#xuất các vị trí bị lặp
A=M5[SS1, mau.flatten()] 
A=A.T
# Tìm các giá trị duy nhất và vị trí của chúng (giữ nguyên thứ tự xuất hiện - stable)
_, w = np.unique(A, return_index=True)
# Sắp xếp chỉ số w để giữ thứ tự xuất hiện ban đầu (stable)
w = np.sort(w)
# Xác định các vị trí của phần tử bị lặp lại
all_indices = np.arange(len(A))  # Index bắt đầu từ 0
dup = np.setdiff1d(all_indices, w)  # Không cần +1 vì index bắt đầu từ 0
dup=dup.reshape(-1, 1) 
mau=mau[dup]

#tính độ chính xác
sizeMau = dup.shape
sz = sizeMau[0] 
dcx = (nj - sz + 1) / nj
dcx = np.array([dcx])
dcx=dcx.reshape(-1, 1) 
dochinhxac = np.concatenate((dochinhxac, dcx), axis=1);

#####################################################################################################################
#####vòng lặp cho thuật toán 
while dcx <= 1:
    SST=SS+1
    giatribilap=np.empty((0, 0))
    entr = np.empty((0, 1))
    A = np.empty((0, 1))
    for i in range(1, nj+1):
        if i not in SST:
            row = M5[i-1, mau.flatten()]
            A = shannon_entropy(row)
        else:
            A=0
        A= np.array([A])
        A = A.reshape(-1, 1) 
        entr=np.concatenate((entr, A), axis=0)
        A = np.empty((0, 1))    

    #tinhMIcuatatcacacnutsovoinutduocchon(tainhugvitribilap)
    #tạo matrix chứa các giá trị bị lặp của các sensor được chọn 
    matrix=np.empty((mau.shape[0], 0))
    MTT=np.empty((mau.shape[0], 0))
    for i in range(1, nj+1):
        if i not in SST:
            continue
        else:  
            MTT=M5[i-1, mau.flatten()]
            MTT=MTT.reshape(-1, 1) 
        matrix=np.concatenate((matrix, MTT), axis=1)


    #mã hóa Matrix
    _, matrix_MH = np.unique(matrix, axis=0, return_inverse=True) 
    matrix_MH = matrix_MH + 1  # Để bắt đầu từ 1 thay vì 0
    matrix_MH=matrix_MH.reshape(-1, 1)

    #tính MI 
    array=np.empty((0, 1))
    MI = np.empty((0, 1))
    MIT = np.empty((0, 1))
    for i in range(1, nj+1):
        array=M5[i-1, mau.flatten()]
        if i not in SST:
            MIT = calculate_featurewise_mutual_information(matrix_MH, array)
        else: 
            MIT=0
        MIT=MIT
        MIT= np.array([MIT])
        MIT= MIT.reshape(-1, 1)
        MI=np.concatenate((MI, MIT), axis=0)
        MIT=np.empty((0, 1))

    #tinhentropychiaMI
    result = np.empty((0, 1))
    for i in range(0, nj):
        if i not in SST:
            T1 = entr[i, 0]
            T2 = MI[i, 0]
            T = T1/T2
            if T2 != 0:
                T = T1 / T2
            else:
                T = 0
        else:
            T=0
        T= np.array([T])
        T= T.reshape(-1, 1)
        result=np.concatenate((result, T), axis=0)

    #lựa chọn nút có kết quả lớn nhất và gán vào nút được chọn
    max_index = np.argmax(result)
    max_index=max_index.reshape(-1, 1)
    SS= np.concatenate((SS, max_index), axis=1);
    SS1=max_index.astype(int).flatten()

    #xuất các vị trí bị lặp
    A=M5[SS1, mau.flatten()] 
    A=A.T
    # Tìm các giá trị duy nhất và vị trí của chúng (giữ nguyên thứ tự xuất hiện - stable)
    _, w = np.unique(A, return_index=True)
    # Sắp xếp chỉ số w để giữ thứ tự xuất hiện ban đầu (stable)
    w = np.sort(w)
    # Xác định các vị trí của phần tử bị lặp lại
    all_indices = np.arange(len(A))  # Index bắt đầu từ 0
    dup = np.setdiff1d(all_indices, w)  # Không cần +1 vì index bắt đầu từ 0
    dup=dup.reshape(-1, 1) 
    mau=mau[dup]

    #tính độ chính xác
    sizeMau = dup.shape
    sz = sizeMau[0] 
    dcx = (nj - sz + 1) / nj
    dcx = np.array([dcx])
    dcx=dcx.reshape(-1, 1) 
    dochinhxac = np.concatenate((dochinhxac, dcx), axis=1);
#####################################################################################################################
print(SS)                       #kiểm tra vị trí sensor
print(dochinhxac)               #kiểm tra độ chính xác
print(entr)                     #kiểm tra kết quả entropy
print(MI[12])                   #kiểm tra kết quả MI
last_element = int(SS[0, -1])
print(last_element)
print(result[last_element])     #kiểm tra kết quả entr/MI
print(dochinhxac*nj-1)          #kiểm tra số nút được bao phủ

