 #!/bin/bash
 t0=$(date +%s%N | cut -b1-13)
 sleep 3
 t1=$(date +%s%N | cut -b1-13)
 
 echo $((t1-t0))
 
