# Datasets

To verify the correctness of the outputs from the optimisation models, it is necessary to run test cases. This folder contains 30 sample datasets to be tested on the optimisation models.

## Dataset naming convention

The datasets are divided into two categories: Randomised and Clustered.

# Randomised
Randomised datasets are those where the zones of interest are equally dispersed across the map. 
For example, orders from zones 1, 10 and 15 are almost equally separated from each other, no distinguishable cluster can be observed.

Example of a randomised dataset: *HL1.csv*

The first alphabet represents the busyness of the depots.

*Low(L) -> 3-4 orders per Depot*

*Medium(M) -> 5-6 orders per Depot*

*High(H) -> 8-9 orders per Depot*

*Extreme(E) -> 10-15 orders per Depot*

The second alphabet represents the type of time window.

*Tight(T) -> 5 min time windows*

*Large(L) -> 30 min time windows*

The digit represents a unique label for each permutation of alphabet pairs (1 or 2).

There are a total of 16 datasets in this category.

# Clustered
Clustered datasets are those where the zones of interest are sufficiently close to each other such that possible clusters can be formed across the map. 

Example of a randomised dataset: *C1.csv*

*C1-C2* features 5 zones that form 2 distinguishable clusters that are far from each other. For example, Cluster (Zones 1-2-3) and Cluster (Zones 14-15).

*C3-C8* features 5 zones that form 2 distinguishable clusters that are close to each other. For example, Cluster (Zones 1-2-3) and Cluster (Zones 8-9).

*C9-C10* features 5-7 zones that form 3 distinguishable clusters. For example, Cluster (Zones 1-2-3), Cluster (Zones 8-9) and Cluster (Zones 15-16).

*C11-C14* features 5 zones that form 1 distinguishable cluster. For example, Cluster (Zones 1-2-3-4-5).

*C1, C3, C5, C7, C9, C11* and *C13* comprises of zones with large time windows (30min) whereas
*C2, C4, C6, C8, C10, C12* and *C14* comprises of zones with tight time windows (5min).

There are a total of 14 datasets in this category.

## Structure of dataset

| Order_ID  | Request_Type | Zone | Demand | Start_TW | End_TW | Port |
| --------- | ------------ |----- | ------ |--------- | ------ | ---- |
| 1  | 1 | 5 | 4 | 540  | 600  | West|
| 2  | 2  | 26 | 2 | 600  | 650  | MSP |

Each dataset contains booking orders (rows), with the following column labels.

Order_ID: *Unique label per booking record*

Request_Type: *Pickup(1), Delivery(2)*

Zone: *Location(Port West, Port MSP, Z1 - Z30)*

Demand: *Number of passengers per request*

Start_TW: *Start of time window, Eg. 540 represents 09:00, where 540 = 9 * 60 minutes*

End_TW: *End of time window*

Port: *Ferry Terminal*

