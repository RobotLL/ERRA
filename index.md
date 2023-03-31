## ERRA: An Embodied Representation and Reasoning Architecture for Long-horizon Language-conditioned Manipulation Tasks

## Code
Code is available [here](https://github.com/RobotLL/ERRA).

## Video

<center> <iframe width="640" height="360" src="https://user-images.githubusercontent.com/32490390/229057136-3816827f-0811-482a-843a-bb01a0c45937.mp4" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center> 

## Examples of Language Instruction in Testing

|     Short-horizon Tasks                                |     Long-horizon   Tasks                                          |
|--------------------------------------------------------|-------------------------------------------------------------------|
|     Grasp an object (i.e., trash)                      |     Please clean the table                                        |
|     Place the object into the bin                      |     Please put sth into the drawer (cosmetic, can, clip)          |
|     Open the drawer                                    |     Please cut sth (eggplant, banana, apple)                      |
|     Close the drawer                                   |     Please put all the round objects from the table to the box    |
|     Place sth into the drawer (cosmetic, can, clip)    |                                                                   |
|     Grasp the knife                                    |     Hybird Tasks                                                  |
|     Cut sth (eggplant, banana, apple)                  |     Please close the drawer and then grasp the knife              |
|     Grasp a round object                               |     Please put sth into the drawer and then clean the table.      |
|     Place the round object into the bin                |     Please clean the table and then cut sth (e.g., banana)        |
|     Grasp sth (cosmetic, can, clip)                    |                                                                   |

## Unseen Verb and Noun

|      Verb|Noun      |
|----------------------------------------|-----------------------------------|
|     put → move, place, pick            |     can → jar,   cola             |
|     cut → chop, slice                  |     cosmetic → makeup             |
|     clean → empty, clear               |     apple, banana→fruit           |
|     close → shut                       |     eggplant → vegetable          |
|     grasp → grip, catch                |     table → tableland,   stage    |
|     open → unclose, unlock             |     object → thing, item          |
|     place → put, set                   |     bin → box, dustbin            |

## Action Language Set

|       |       |
| :----------------------------- | :---------------------------------------- |
| Grasp an object                | Grasp the + “object name”                 |
| Place the object into the bin  | Cut the + “object name”                   |
| Open the drawer                | Place the + “object name” into the drawer |
| Close the drawer               | Grasp a round object                      |
| Done                           | Place the round object into the bin       |


  
