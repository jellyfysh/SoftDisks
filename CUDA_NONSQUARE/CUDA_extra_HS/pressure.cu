//
//   Copyright (c) 2023, Yoshihiko Nishikawa, Werner Krauth, and A. C. Maggs
//
//   CUDA code for massively parallelized Monte Carlo simulation of
//   two-dimensional disks
//
//   URL: https://github.com/jellyfysh/SoftDisks
//   See LICENSE for copyright information
//
//   If you use this code or find it useful, please cite the following paper:
//
//   @article{PhysRevE.108.024103,
//       title = {Liquid-hexatic transition for soft disks},
//       author = {Nishikawa, Yoshihiko and Krauth, Werner and Maggs, A. C.},
//       journal = {Phys. Rev. E},
//       volume = {108},
//       issue = {2},
//       pages = {024103},
//       numpages = {7},
//       year = {2023},
//       month = {Aug},
//       publisher = {American Physical Society},
//       doi = {10.1103/PhysRevE.108.024103},
//       url = {https://link.aps.org/doi/10.1103/PhysRevE.108.024103}
//   }
//
//
#include "../CUDA_central/Soft.h"
#include <fstream>
#include "../CUDA_central/fastpow.h"


void ecmc_measure_pressure(double & outputp)
{


  double average = 0.0;
  int max_count = 10; // Number of event chains to measure pressure
  for(int count = 0; count < max_count; count++){
    int x_index = int(Nblock * genrand(mt));
    int y_index = int(Nblock * genrand(mt));
    while(nparticle(x_index, y_index) == 0)
      x_index = int(Nblock * genrand(mt)), y_index = int(Nblock * genrand(mt));
  
    int p_index = int(nparticle(x_index, y_index) * genrand(mt));

    int next_x_index, next_y_index, next_p_index;

    double max_length = Lblockx * 10;
    double cumulated_length = 0;
    double excess = 0;
    while(cumulated_length < max_length){
      double x0 = rx(x_index, y_index, p_index);
      double y0 = ry(x_index, y_index, p_index);
      double min_dist = 10.0 * Lblockx;
      next_x_index = x_index;
      next_y_index = y_index;
      next_p_index = p_index;
      short flag_found = 0;
      double lift = 0;

      // Find a candidate within the same cell to which the moving particle collides
      for(int m = 0; m < nparticle(x_index, y_index); m++){ 
      	if(m != p_index){
      	  double x1 = rx(x_index, y_index, m);
    	    double y1 = ry(x_index, y_index, m);
    	    double dist_x = x1 - x0;
      	  double dist_y = y1 - y0;
    	    if(fabs(dist_y) < 1.0 && dist_x > 0.0){
    	      double togo = dist_x - sqrt(1.0 - dist_y * dist_y);
    	      if(togo < min_dist){
      	      min_dist = togo;
    	        next_p_index = m;
    	        flag_found = 1;
    	        lift = sqrt(1.0 - dist_y * dist_y);
      	    }
    	    }
      	}
      }
      
      // Find a candidate to which the moving particle collides in difference cells
      int x = 0;
      do {
      	for(int y = -1; y <= 1; y++){
      	  if(x != 0 || y != 0){
      	    for(int m = 0; m < nparticle((x_index + x + Nblock) % Nblock, (y_index + y + Nblock) % Nblock); m++){
      	      double x1 = rx((x_index + x + Nblock) % Nblock, (y_index + y + Nblock) % Nblock, m) + (double)x * Lblockx;
      	      double y1 = ry((x_index + x + Nblock) % Nblock, (y_index + y + Nblock) % Nblock, m) + (double)y * Lblocky;

      	      double dist_x = x1 - x0;
      	      double dist_y = y1 - y0;
      	      if(fabs(dist_y) < 1.0 && dist_x > 0.0){
            		double togo = dist_x - sqrt(1.0 - dist_y * dist_y);
            		if(togo < min_dist){
            		  min_dist = togo;
            		  next_x_index = (x_index + x + Nblock) % Nblock;
            		  next_y_index = (y_index + y + Nblock) % Nblock;
            		  next_p_index = m;
            		  flag_found = 1;
            		  lift = sqrt(1.0 - dist_y * dist_y);
            		}
      	      }
      	    }
      	  }
      	}
      	x++;
      } while (flag_found == 0 || x < 2);
    
      if(cumulated_length + min_dist > max_length){
      	rx(x_index, y_index, p_index) += max_length - cumulated_length;
      	lift = 0;
      } else {
      	rx(x_index, y_index, p_index) += min_dist;
      	excess += lift;
      }

      if(rx(x_index, y_index, p_index) > Lblockx){
      	move_block(x_index, y_index, p_index);
      	index_shift(x_index, y_index, p_index);
      }

      cumulated_length += min_dist;
      x_index = next_x_index;
      y_index = next_y_index;
      p_index = next_p_index;
    }
    average += excess / max_length;
  }
  outputp = density * (1.0 + average / (double)max_count);


}

void measure_pressure(double & outputpx, double & outputpy)
{
  int allsize = Nblock * Nblock * nmax;
  vector <vector <int>> cindex(allsize, vector<int>(6));
  vector <vector <double>> cdistance(allsize, vector<double>(4));

  find_first_collision(cindex, cdistance);

  double average_x = 0.0, average_y = 0.0;
  double chain_length = 0.1;
  for (int x_index = 0; x_index < Nblock; x_index++){
    for(int y_index = 0; y_index < Nblock; y_index++){
      for(int p_index = 0; p_index < nparticle(x_index, y_index); p_index++){
        int index = p_index * Nblock * Nblock + y_index * Nblock + x_index;

        double temp_chain_length_x = chain_length;
        double excess_x = 0;

        while(temp_chain_length_x > cdistance[index][0]){ 
          temp_chain_length_x -= cdistance[index][0];
          excess_x += cdistance[index][1];
          index = cindex[index][2] * Nblock * Nblock + cindex[index][1] * Nblock + cindex[index][0];
        }
        average_x += excess_x / chain_length;

        double temp_chain_length_y = chain_length;
        double excess_y = 0;
        index = p_index * Nblock * Nblock + y_index * Nblock + x_index;
        while(temp_chain_length_y > cdistance[index][2]){ 
          temp_chain_length_y -= cdistance[index][2];
          excess_y += cdistance[index][3];
          index = cindex[index][5] * Nblock * Nblock + cindex[index][4] * Nblock + cindex[index][3];
        }
        average_y += excess_y / chain_length;

      }
    }
  }

  outputpx = density * (1.0 + average_x / (double)NNN);
  outputpy = density * (1.0 + average_y / (double)NNN);

}


void find_first_collision(vector <vector <int>> & cindex, vector < vector <double>> & cdistance)
{

  for (int x_index = 0; x_index < Nblock; x_index++){
    for(int y_index = 0; y_index < Nblock; y_index++){
      for(int p_index = 0; p_index < nparticle(x_index, y_index); p_index++){
        int next_p_index = p_index, next_x_index = x_index, next_y_index = y_index;
        int next_p_indey = p_index, next_x_indey = x_index, next_y_indey = y_index;

        double x0 = rx(x_index, y_index, p_index);
        double y0 = ry(x_index, y_index, p_index);

        double min_dist_x = 10.0 * Lblockx, min_dist_y = 10.0 * Lblocky;
        double excess_x, excess_y;
        bool flag_found_x = false, flag_found_y = false;

        // Find a candidate within the same cell to which the moving particle collides

        for(int m = 0; m < nparticle(x_index, y_index); m++){ 
          if(m != p_index){
            double x1 = rx(x_index, y_index, m);
            double y1 = ry(x_index, y_index, m);
            double dist_x = x1 - x0;
            double dist_y = y1 - y0;

            // in the +x direction
            if(fabs(dist_y) < 1.0 && dist_x > 0.0){
              double togo = dist_x - sqrt(1.0 - dist_y * dist_y);
              if(togo < min_dist_x){
                min_dist_x = togo;
                next_p_index = m;
                flag_found_x = true;
                excess_x = sqrt(1.0 - dist_y * dist_y);
              }
            }
            // in the +y direction
            if(fabs(dist_x) < 1.0 && dist_y > 0.0){
              double togo = dist_y - sqrt(1.0 - dist_x * dist_x);
              if(togo < min_dist_y){
                min_dist_y = togo;
                next_p_indey = m;
                flag_found_y = true;
                excess_y = sqrt(1.0 - dist_x * dist_x);
              }
            }

          }
        }
    
        // Find a candidate to which the moving particle collides in difference cells
        
        {// in the x direction
      	  int x = 0;
      	  do {
      	    for(int y = -1; y <= 1; y++){
      	      if(x != 0 || y != 0){
            		for(int m = 0; m < nparticle((x_index + x + Nblock) % Nblock, (y_index + y + Nblock) % Nblock); m++){
          	  	  double x1 = rx((x_index + x + Nblock) % Nblock, (y_index + y + Nblock) % Nblock, m) + (double)x * Lblockx;
            		  double y1 = ry((x_index + x + Nblock) % Nblock, (y_index + y + Nblock) % Nblock, m) + (double)y * Lblocky;
		  
            		  double dist_x = x1 - x0;
            		  double dist_y = y1 - y0;
            		  if(fabs(dist_y) < 1.0 && dist_x > 0.0){
            		    double togo = dist_x - sqrt(1.0 - dist_y * dist_y);
            		    if(togo < min_dist_x){
            		      min_dist_x = togo;
            		      next_x_index = (x_index + x + Nblock) % Nblock;
            		      next_y_index = (y_index + y + Nblock) % Nblock;
            		      next_p_index = m;
            		      flag_found_x = true;
            		      excess_x = sqrt(1.0 - dist_y * dist_y);
            		    }
            		  }
            		}
      	      }
      	    }
      	    x++;
      	  } while (flag_found_x == false || x < 2);
      	  int index = p_index * Nblock * Nblock + y_index * Nblock + x_index;
      	  cindex[index][0] = next_x_index, cindex[index][1] = next_y_index, cindex[index][2] = next_p_index;
      	  cdistance[index][0] = min_dist_x;
      	  cdistance[index][1] = excess_x;
        }

        {// in the y direction
          int y = 0;
      	  do {
      	    for(int x = -1; x <= 1; x++){
      	      if(x != 0 || y != 0){
            		for(int m = 0; m < nparticle((x_index + x + Nblock) % Nblock, (y_index + y + Nblock) % Nblock); m++){
          		    double x1 = rx((x_index + x + Nblock) % Nblock, (y_index + y + Nblock) % Nblock, m) + (double)x * Lblockx;
  		            double y1 = ry((x_index + x + Nblock) % Nblock, (y_index + y + Nblock) % Nblock, m) + (double)y * Lblocky;

                  double dist_x = x1 - x0;
            		  double dist_y = y1 - y0;
            		  if(fabs(dist_x) < 1.0 && dist_y > 0.0){
            		    double togo = dist_y - sqrt(1.0 - dist_x * dist_x);
            		    if(togo < min_dist_y){
            		      min_dist_y = togo;
          	  	      next_x_indey = (x_index + x + Nblock) % Nblock;
          		        next_y_indey = (y_index + y + Nblock) % Nblock;
          		        next_p_indey = m;
          		        flag_found_y = true;
          		        excess_y = sqrt(1.0 - dist_x * dist_x);
          		      }
            		  }
            		}
      	      }
    	      }
    	      y++;
  	      } while (flag_found_y == false || y < 2);
      	  int index = p_index * Nblock * Nblock + y_index * Nblock + x_index;
      	  cindex[index][3] = next_x_indey, cindex[index][4] = next_y_indey, cindex[index][5] = next_p_indey;
      	  cdistance[index][2] = min_dist_y;
      	  cdistance[index][3] = excess_y;
        }
      }
    }
  }

}

void move_block(int x_index, int y_index, int p_index)
{

  int num_block = int(rx(x_index, y_index, p_index) / Lblockx);
  int target_index = (x_index + num_block) % Nblock;

  rx(target_index, y_index, nparticle(target_index, y_index)) 
    = rx(x_index, y_index, p_index) - num_block * Lblockx;
  ry(target_index, y_index, nparticle(target_index, y_index)) 
    = ry(x_index, y_index, p_index);
  
  nparticle(target_index, y_index)++;

  for(int i = 0; i < nparticle(target_index, y_index) - 1; i++){
    double dist = (rx(target_index, y_index, i) - rx(target_index, y_index, nparticle(target_index, y_index)))
      * (rx(target_index, y_index, i) - rx(target_index, y_index, nparticle(target_index, y_index)))
      + (ry(target_index, y_index, i) - ry(target_index, y_index, nparticle(target_index, y_index)))
      * (ry(target_index, y_index, i) - ry(target_index, y_index, nparticle(target_index, y_index)));
    if(dist < 1.0){
      cerr << rx(x_index, y_index, p_index) << " " << num_block << " " << Lblockx << endl;
    }

  }

}

void index_shift(int x_index, int y_index, int p_index)
{

  if(p_index < nparticle(x_index, y_index) - 1){
    for(int i = p_index + 1; i < nparticle(x_index, y_index); i++){
      rx(x_index, y_index, (i - 1)) = rx(x_index, y_index, i);
      ry(x_index, y_index, (i - 1)) = ry(x_index, y_index, i);
    }
  }
  rx(x_index, y_index, (nparticle(x_index, y_index) - 1)) = DUMMY_OFFSET;
  ry(x_index, y_index, (nparticle(x_index, y_index) - 1)) = DUMMY_OFFSET;
  nparticle(x_index, y_index)--;
}


void check_particle_number(void)
{

  int sum_particle = 0;
  for(int i = 0; i < Nblock * Nblock; i++){
    if(nparticle[i] >= nmax){
      cerr << "ERROR " << nparticle[i] << endl;
      assert(nparticle[i] < nmax);
    }
    sum_particle += nparticle[i];
  }
  
  if(sum_particle != NNN)
    cerr << "Error " << sum_particle << " != " << NNN << endl, assert(sum_particle == NNN);

}
