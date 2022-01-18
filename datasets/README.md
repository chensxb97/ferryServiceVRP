# Datasets

This folder contains sample datasets that represent test cases for the optimisation problem.

## Structure of a dataset

Each dataset contains booking orders (rows), with the following column labels.

#### Order_ID: 
Unique label per booking record

#### Request_Type: 
(1) for Pickup, (2) for Delivery

#### Zone: 
Unique number that represents each zone location, Z1 - Z30

#### Demand:
Number of passengers per request

#### Start_TW:
Start of time window
Eg. 540 represents 09:00, where 540 = 9 * 60 minutes

#### End_TW:
End of time window

#### Port:
Ferry Terminal

| Order_ID  | Request_Type | Zone | Demand | Start_TW | End_TW | Port |
| --------- | ------------ |----- | ------ |--------- | ------ | ---- |
| 1  | 1 | 5 | 4 | 540  | 600  | West|
| 2  | 2  | 26 | 2 | 600  | 650  | MSP |


