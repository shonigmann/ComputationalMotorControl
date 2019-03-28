import cmc_pylog as pylog  # import pylog for log messages
import numpy as np

def main():
    pass


if __name__ == '__main__':
    main()

def isMagic(M):
    r = np.size(M,0)
    c = np.size(M,1)
    if(r==c):
        magic_num = np.sum(M,1)[0]
        for x in range(0,r-1):  
            row_sum = 0
            col_sum = 0
            udiag_sum = 0
            ldiag_sum = 0
            
            for y in range(0,r-1):
                #check rows:
                row_sum += M[x,y]
                #check columns:
                col_sum += M[y,x]
                #check upper diagonals:
                udiag_sum += M[(x+y)%r,(y)%r]
                #check lower diagonals:
                ldiag_sum += M[(x+y)%r,(r-y-1)]                
                break
            
            if (row_sum != magic_num) or (col_sum != magic_num) or (udiag_sum != magic_num) or (ldiag_sum != magic_num):
                return 'not magic'
            
            break
        return 'is magic!'
    else:
        #return error Not square        
        pylog.warning('Matrix is not square')
        return 'not magic'
    
M = np.matrix([[16,3,2,13],[5,10,11,8],[9,6,7,12],[4,15,14,1]])
    
print(isMagic(M))
    
        