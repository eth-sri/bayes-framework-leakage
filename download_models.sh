wget -r --no-parent https://files.sri.inf.ethz.ch/bayes-framework-leakage/
find ./files.sri.inf.ethz.ch/bayes-framework-leakage/ -name "index.html" -print0 | xargs -0 rm -rf
rsync -av files.sri.inf.ethz.ch/bayes-framework-leakage/* ./
rm -rf files.sri.inf.ethz.ch/
