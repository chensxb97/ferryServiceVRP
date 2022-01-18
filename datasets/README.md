# Datasets

This folder contains sample datasets that represent test cases for the optimisation problem.

## Category naming convention

Example of a dataset: *[L][H][1].csv*

The first alphabet represents the type of time window.

*Tight(T) -> 5 min time windows*

*Moderate(M) -> 15 min time windows*

*Large(L) -> 30 min time windows*

The second alphabet represents the busyness of the orders.

*Low(L) -> 3 orders per Depot*

*Medium(M) -> 6 orders per Depot*

*High(H) -> 9 orders per Depot*

The digit is a unique label per dataset category.
There are 5 datasets per category, giving rise to 45 datasets.

The example csv file represents a dataset that features a higher number of orders per depot with large time windows.

## Structure of a dataset

| Order_ID  | Request_Type | Zone | Demand | Start_TW | End_TW | Port |
| --------- | ------------ |----- | ------ |--------- | ------ | ---- |
| 1  | 1 | 5 | 4 | 540  | 600  | West|
| 2  | 2  | 26 | 2 | 600  | 650  | MSP |

Each dataset contains booking orders (rows), with the following column labels.

Order_ID: *Unique label per booking record*

Request_Type: *(1) for Pickup, (2) for Delivery*

Zone: *Unique number that represents each zone location, Z1 - Z30*

Demand: *Number of passengers per request*

Start_TW: *Start of time window, Eg. 540 represents 09:00, where 540 = 9 * 60 minutes*

End_TW: *End of time window*

Port: *Ferry Terminal*

