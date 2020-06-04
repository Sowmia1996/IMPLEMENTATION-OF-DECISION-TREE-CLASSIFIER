#Assumptions i have made:
#class column is the last column
#the program trains with training-data.csv and tests with testingdata.csv
#I am still working on making the code work for categorical attributes ,so this code is only working for numerical attributes
import math
import pandas as pd
import operator
import itertools
#making a data frame
df=pd.read_csv('training-data.csv')
print(df)


                     ##  function to return all the attributes sorted independently   ##
        ##  each sorted attribute list has entries with two fields -{<attribute value,index into classlist>} ##
def pre_sorting_attributes(df):
    
    attributes=df.columns.tolist()
    attributetype=attribute_type(attributes)
    loa=len(attributes)
    sorted_attributes=[[] for i in range(loa-1)]
    for k in range(loa-1):
        if(attributetype[k]==0):                 #if numerical attribute then sort it
            for i in range (len(df)):
                row=[df[attributes[k]][i],i]
                sorted_attributes[k].append(row)
            sorted_attributes[k]=sorted(sorted_attributes[k],key=operator.itemgetter(0))
            
            
        if(attributetype[k]!=0):              #if categorical attribute,then find all possible subsets of unique elements in the attribute
            sorted_attributes[k]=list(df[attributes[k]].unique())
    return sorted_attributes



                          ## class to construct the nodes of the decision tree##
class node():

    def __init__(self,label="root",histogram=[],left=None,right=None):
        self.identity=label
        self.classfreq=histogram # i need this attribute to check if the node is pure to prevent frm further partiotioning
        self.left=left
        self.right=right
        self.best_split=0
        
    def display_node(self):
        print("node name:",self.identity,"\nclass_frequency for the node:",self.classfreq)
        
        
                 # method creates a left child for the node and returns the child#
    def create_leftchild(self,label,histogram):
        child=node(label,histogram)
        self.left=child
        return child
    
                 # method creates a right child for the node and returns the child#     
    def create_rightchild(self,label,histogram):
        child=node(label,histogram)
        self.right=child
        return child
        
                # method checks if the node is a pure node and returns 1 if pure and 0 if not pure #
    def check_ifleaf(self):
        if(self.left==None and self.right==None):
            return 1
        else:
            return 0
        
                 # method that sets the best_split for numerical attribute of node to the value passed#
    def set_best_split(self,best_split):
        self.best_split=best_split
        
             # method that sets the best_split for categorical attribute pf node  to the value passed#
    def set_best_split_cat(self,best_split):
        self.best_split_cat=best_split
        
                #method that returns the best_split(numerical att)for the current  node#
    def get_best_split(self):
        return(self.best_split)
    
                 #method that returns the best_split(categorical att) for the current node#
    def get_best_split_cat(self):
        return self.best_split_cat
            
                 # method checks if it is a pure node and returns the class type  for the pure node   
    def get_pure_class(self):            
        from collections import Counter
        histogram=self.classfreq
        return([k for k,v in histogram.most_common()][0])
    
    
                # function updating initial classlist#
def update_classlist(df,leaf=None):
    classlist=[]
    attributes=df.columns.tolist()
    noa=len(attributes)
    for i in range(len(df)):
        row=[df[attributes[noa-1]][i],leaf]
        classlist.append(row)
    return classlist


            #function updating a partiton of classlist#
def update_classlist2(partition,nodes):
    for k in range(len(partition)):
        partition[k][1]=nodes
    return partition


            # function for getting histogram(class frequency disribution for a classlist #
def get_histogram(classlist):
    from collections import Counter
    histogram=Counter(classlist[i][0] for i in range(len(classlist)))
    return histogram

            # creating root node and updating its histogram
root=node("root",get_histogram(update_classlist(df)))
update_classlist(df,root)

            
            # function returning gini for the root node(if its pure .. we never do any partition)
def gini_rootnode(classlist):
    gini=1
    histogram_root=get_histogram(classlist)
    gini-=sum(map(lambda x:x*x,[(histogram_root[i])/len(classlist) for i in histogram_root]))
    return gini
 
    
                          # function to return gini for a split
                 #receives classlist for 2 partitions  as arguements
def gini_calculate(p1,p2):
    gini_1=1
    gini_2=1
    histogram_p1=get_histogram(p1)
    histogram_p2=get_histogram(p2)
    gini_1 -= sum(map(lambda x:x*x,[(histogram_p1[i])/len(p1) for i in histogram_p1]))
    gini_2 -= sum(map(lambda x:x*x,[(histogram_p2[i])/len(p2) for i in histogram_p2]))
    gini=(len(p1)*(gini_1)/(len(p1)+len(p2)))+(len(p2)*(gini_2)/(len(p1)+len(p2)))
    #print(gini)
    return [gini,gini_1,gini_2]


                         # function that receives indices into a partiton and returns the next  sorted attribute with those indices#
def create_sublist(indices,q,attlist):
    list1=[]
    list2=[attlist[q][k][1] for k in range(len(df))]
    for i in indices:
        list1.append(list2.index(i))
    list1=sorted(list1)
    return([attlist[q][k] for k in list1])

#function to identify whether categorical or numerical -returns 1 if categorical and 0 if numerical
def attribute_judge(q):
    attributes=df.columns.tolist()
    k=attributes[q].find(':c')
    l=attributes[q].find(':n')
    return (k>l)



#function that returns all subsets of a set without its compliment
import itertools
from itertools import chain,combinations
def get_subsets(attlist):
    k=list(chain.from_iterable(list(combinations(attlist,n) for n in range(1,int(len(attlist)/2)+2))))
    return k 




                 #function that recursively partitions for categorical attributes(can switch from numerical to categorical and vice-versa)
def CAT_PARTITION(q,index,nodes,attribute_list):
    classlist=update_classlist(df)
    attributes=df.columns.tolist()
    attlist=[attribute_list[q][k] for k in index ]
    a=[attlist[k][0] for k in range(len(attlist))]
    c=list(set(a)) #unique elements in a
    subsets=get_subsets(c) # returns only list with subsets whose complimant is not there!!
    minimum_gini=math.inf
    best_split=0
    best_P1=[]
    best_P2=[]
    for i in range(len(subsets)): 
        list1=[]
        for j in range(len(subsets[i])):
            list1.append([k for k,val in enumerate(a) if val==subsets[i][j]])
        list1=[item for sublist in list1 for item in sublist]
        b=[attlist[k][1] for k in range(len(attlist))]
        list2=list(set(b).difference(attlist[k][1] for k in list1))
        partition_1=[classlist[j] for j in [attlist[k][1] for k in list1]]
        partition_2=[classlist[j] for j in list2]
        GINI=gini_calculate(partition_1,partition_2)
        gini_split=GINI[0]
        if(gini_split< minimum_gini):
            minimum_gini=gini_split
            best_split=i
            best_P1=list1
            best_P2=list2
    print("best split happens at", attributes[q],"=",subsets[best_split])
    
    
    # best PARTITIONS to update histogram
    P1=[classlist[j] for j in [attlist[k][1] for k in best_P1]]
    P2=[classlist[j] for j in best_P2]
    BEST_GINI=gini_calculate(P1,P2)
    
    
    #CREATE A LEFT CHILD NODE
    #child1=node(str(attributes[q-1])+'<'+str(attlist[best_split][0]),get_histogram(P1))
    child1=nodes.create_leftchild(str(subsets[best_split]),get_histogram(P1))  # link child1 to nodes
    nodes.set_best_split_cat(subsets[best_split])
    child1.display_node()
    update_classlist2(P1,child1) #updating classlist
    
    
    #CREATE A RIGHT CHILD NODE 
    #child2=node(str(attributes[q-1])+'>'+str(attlist[best_split][0]),get_histogram(P2))
    child2=nodes.create_rightchild(str(set(c).difference(subsets[best_split])),get_histogram(P2))              #link child2 tonodes
    nodes.set_best_split_cat(subsets[best_split])
    child2.display_node()
    update_classlist2(P2,child2) #updating classlist
                                   
    q+=1 #moving on to next categorical attribute                                
    if(BEST_GINI[1]!=0): #if P1 is not a pure node
        CAT_PARTITION(q,list(attlist[k][1] for k in best_P1),child1,attribute_list)
                    
                    
    if(BEST_GINI[2]!=0): #if P2 is not a pure node
        CAT_PARTITION(q,best_P2,child2,attribute_list)
                    

        







                       # function that recursively partitions the node untill all sub-nodes become pure
def PARTITION(attlist,nodes,q,attribute_list):
            attributes=df.columns.tolist()
            classlist=update_classlist(df)
            minimum_gini=math.inf     # at each split minimum_gini gets updated ..at end if att1 traversl we have best split
            best_split=0            # at end of traversal of att1 .. best_split has  where the split has to happen
            for i in range(len(attlist)-1):
                partition_1=[classlist[j]for j in list(attlist[k][1] for k in list(range(0,i+1)))]
                partition_2=[classlist[j]for j in list(attlist[k][1] for k in list(range(i+1,len(attlist))))]
                GINI=gini_calculate(partition_1,partition_2)
                gini_split=GINI[0]
                if(gini_split< minimum_gini):
                    minimum_gini=gini_split
                    best_split=i                          #minimum_gini -best split for attribute1
         
            print("best split happens at", attributes[q],"=",attlist[best_split][0])
            P1=[classlist[j] for j in [a[1] for a in attlist[0:best_split+1]]]              #partitions created for the best split
            P2=[classlist[j] for j in [a[1] for a in attlist[best_split+1:len(attlist)]]]
            BEST_GINI=gini_calculate(P1,P2)            #calculating gini for the split
            index2=[attlist[k][1] for k in list(range(best_split+1,len(attlist)))]  
            index1=[attlist[k][1] for k in range(best_split+1)]
            
            q+=1     #move on to next attribute
            
            #CREATE A LEFT CHILD NODE
            child1=nodes.create_leftchild((str(attributes[q-1])+'<'+str(attlist[best_split][0])),get_histogram(P1))                         
            nodes.set_best_split(attlist[best_split][0])
            child1.display_node()
            update_classlist2(P1,child1) #updating classlist
            
            #CREATE A RIGHT CHILD NODE 
            child2=nodes.create_rightchild((str(attributes[q-1])+'>'+str(attlist[best_split][0])),get_histogram(P2))                       
            nodes.set_best_split(attlist[best_split][0])
            child2.display_node()
            update_classlist2(P2,child2) #updating classlist
            
            
            
            
                
                 # P1 IS NOT A PURE NODE!!
            if(BEST_GINI[1]!=0):    
                attlist1=create_sublist(index1,q,attribute_list)
                PARTITION(attlist1,child1,q,attribute_list)                  #partition  wrt next attribute

            
                # P2 IS NOT A PURE NODE!!
            if(BEST_GINI[2] !=0):    
                
                attlist2=create_sublist(index2,q,attribute_list)
                PARTITION(attlist2,child2,q,attribute_list)                  #partition  wrt next attribute
                

                
                # function that send the PARTITION with initial arguements and sets off the recursion
def getsplit (df):
    #attributes=df.columns.tolist()
    attlist=pre_sorting_attributes(df)
    classlist=update_classlist(df)
    if(gini_rootnode(classlist)==0):
        return
    q=0         #initially doing split for att1 
    PARTITION([attlist[q][k] for k in range(len(df))],root,q,attlist)

getsplit(df)  #calling the function

                   
               # function that traverses the tree according to the inputlist provided and returns the class-type for the imputlist
def traveltree(inputlist):
    travelptr=root              # node is root... travelptr is initially pointing to root node
    i=0
    while(travelptr.right !=None): # if node is not pure..then do:
            if(inputlist[i] < travelptr.get_best_split()):
                travelptr=travelptr.left
            else:
                travelptr=travelptr.right
            i+=1
    # when travelptr is pointing to a pure node.. print the class type of the node 
    print("the data entered has class type:",travelptr.get_pure_class())
    
traveltree([74,50,60,"rich"])    #just testing with some random list    
traveltree([60,80,50,"poor"])
    
    
    
    
                               ## FINALLY!!!  testing the decision-tree classifier with testing-data.csv file
import csv
def train():
    import csv
    with open ('testing-data.csv','r') as csv_file:
        reader =csv.reader(csv_file)
        next(reader)             # skip first row
        for row in reader:
            test_row = [float(i) for i in row] 
            print(test_row)
            traveltree(test_row)

            
            
            
train()
    
    

