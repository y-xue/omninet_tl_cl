seq_lst = ['1-0-0', '0-1-0', '0-0-1',
           '1-1-0', '0-2-0', '2-0-0', '1-0-1', '0-1-1',
           '1-2-0', '2-1-0', '1-1-1', '0-2-1', '2-0-1',
           '2-2-0', '1-2-1', '2-1-1', 
           '2-2-1']
coef_lst = [1,  0.95,0.9,   
            0.8,0.78,0.76,0.74,0.72,      
            0.5,0.48,0.46,0.44,0.42,            
            0.3,0.28,0.25,         
            0.1]
coef_ITTIV = dict(zip(seq_lst, coef_lst))


seq_lst1 = ['I????',
            'IT???', '?T???',
            'I?T??', 'ITT??', '?TT??', '??T??',
            'I??I?', 'IT?I?', '?T?I?', 'I?TI?', 'ITTI?', '?TTI?', '??TI?', '???I?',
            'IT??V', 'I?T?V', '?T?IV', 'ITTIV', '??TIV', '????V']
coef_lst1 = [1,  
            0.8, 0.95,
            0.8, 0.5, 0.78, 0.95,
            0.76, 0.48, 0.8, 0.48, 0.3, 0.5, 0.8, 1,
            0.46, 0.46, 0.46, 0.1, 0.46, 0.9]
coef_ITTIV1 = dict(zip(seq_lst1, coef_lst1))

coef_lst2 = [[1],  
            [0.8]*2,
            [0.5]*4,
            [0.3]*8,
            [0.1]*6]
coef_lst2 = [x for l in coef_lst2 for x in l]
coef_ITTIV2 = dict(zip(seq_lst1, coef_lst2))


def reward_fun(a, x):
    return a*x**10

def reward_ITTIV(seq, acc, count=True):
    if count:
        return reward_fun(coef_ITTIV[seq], acc)
    return reward_fun(coef_ITTIV1[seq], acc)

def reward_ITTIV_t(seq, acc, count=True):
    return reward_fun(coef_ITTIV2[seq], acc)

