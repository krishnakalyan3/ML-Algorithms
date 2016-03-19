INPUTDATE=`date`
git pull
git add .
git commit -m "Commit Date $INPUTDATE"
git remote add origin_ml https://github.com/krishnakalyan3/ML-Algorithms
git push -u origin_ml master
