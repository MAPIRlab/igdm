git clone --recursive https://github.com/MAPIRlab/igdm.git

<!--------------------------------------+-------------------------------------->
#                  Information-Driven Gas Distribution Mapping
<!--------------------------------------+-------------------------------------->

We contribute with an efficient exploration algorithm for 2D gas-distribution mapping with an autonomous mobile robot. Our proposal combines a Gaussian Markov random field estimator based on gas and wind measurements that is devised for very sparse sample sizes and indoor environments, with a partially observable Markov decision process to close the robot's control loop.




<!--------------------------------------+-------------------------------------->
#                                  Example Code
<!--------------------------------------+-------------------------------------->

The simplest starting point is running our test example as follows:
```
python igdm/pomdp/test.py
```
It will run a simulated exploration of a indoor-like environment and report both
to terminal as well as to disk. 





<!--------------------------------------+-------------------------------------->
#                                   Contribute
<!--------------------------------------+-------------------------------------->

This project is only possible thanks to the effort of many, including mentors,
innovators, developers, and of course, our beloved coffe vending machine.
If you like this project and want to contribute in any way, you are most welcome.

Yoy can find a detailed list of everyone involved involved in the development of
this software in [AUTHORS.md](AUTHORS.md). Thanks to all of you!



<!--------------------------------------+-------------------------------------->
#                                    License
<!--------------------------------------+-------------------------------------->

This software was developed in a collaboration between the Machine Perception 
and Intelligent Robotics (MAPIR) research group, University of Malaga, and the 
Distributed Intelligent Systems and Algorithms Laboratory (DISAL) research
group, ??cole Polytechnique F??d??rale de Lausanne (EPFL).

* This software is released under a GPLv3 license.
  Read [license-GPLv3.txt](LICENSE),
  or if not present, <http://www.gnu.org/licenses/>.

* If you need a closed-source version of this software
  for commercial purposes, please contact the authors.

* If you use this software in an academic work,
  please cite the most relevant publication associated
  by reading the individual README files of each technique
  (some are implementation of GDM techniques proposed by third parties)
  or visiting http://mapir.uma.es,
  or if any, please cite the authors of the software directly.
