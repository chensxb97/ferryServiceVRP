# Datasets

This folder contains sample datasets that represent test cases for the optimisation problem.

## Dataset naming convention

Example of a dataset: *HL1.csv*

The first alphabet represents the busyness of the depots.

*Low(L) -> 3-4 orders per Depot*

*Medium(M) -> 5-6 orders per Depot*

*High(H) -> 8-9 orders per Depot*

The second alphabet represents the type of time window.

*Tight(T) -> 5 min time windows*

*Moderate(M) -> 15 min time windows*

*Large(L) -> 30 min time windows*

The digit is a unique label per dataset category.
There are 3 datasets per category, hence, a total of 27 datasets for 9 different categories.

HL1.csv is one of three datasets characterised with a High number of orders per depot with Large time windows.

## Structure of dataset

| Order_ID  | Request_Type | Zone | Demand | Start_TW | End_TW | Port |
| --------- | ------------ |----- | ------ |--------- | ------ | ---- |
| 1  | 1 | 5 | 4 | 540  | 600  | West|
| 2  | 2  | 26 | 2 | 600  | 650  | MSP |

Each dataset contains booking orders (rows), with the following column labels.

Order_ID: *Unique label per booking record*

Request_Type: *Pickup(1), Delivery(2)*

Zone: *location(Port West, Port MSP, Z1 - Z30)*

Demand: *Number of passengers per request*

Start_TW: *Start of time window, Eg. 540 represents 09:00, where 540 = 9 * 60 minutes*

End_TW: *End of time window*

Port: *Ferry Terminal*

