# ML_DIMENSION-REDUCTION
PCA and FastMap Algorithm 
PCA
Use PCA to reduce the dimensionality of the data points in pca-data.txt from 3D to 2D. 
Each line of the data file represents the 3D coordinates of a single point. 
output the directions of the first two principal components.
FastMap 
Use FastMap to embed the objects in fastmap-data.txt into a 2D space. 
The first two columns in each line of the data file represent the IDs of the two objects;
and the third column indicates the symmetric distance between them. 
If the furthest pair of objects is not unique, use the one that includes the smallest object ID. 
After selecting the furthest pair, when computing the coordinates,always use the object with the smaller ID of the pair as the point of coordinate zero.
The objects listed in fastmap-data.txt are actually the words in fastmap-wordlist.txt
(nth word in this list has an ID value of n) and the distances between each pair of objects are the Damerau– Levenshtein distances between them. 
plot the words onto a 2D plane using FastMap solution :please see screen shot:

the implement source code file is: FastMap.py

The detail discussion of algorithms , implementation and library function please see the DIMENSION REDUCTIONreport.pdf.

