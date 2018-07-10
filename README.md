# variantNN
Single Nucleotide Polymorphism (SNP) detection for Single Molecule DNA Sequencing.  <br/>
Reassemble sequence from short reads and call variants.<br/>
<br/>
Dataset from aggregated alignment BAM file:<br/>
    NA12878 PacBio read dataset generated by WUSTL (https://www.ncbi.nlm.nih.gov//bioproject/PRJNA323611)<br/>
    Alignment information stored in 3- 15x4 matrices (15 nt, 4 bases coded as one-hot-coding)<br/>
    1- baseline encoded refernce sequence and total counts<br/>
    2- difference between reference counts and seq counts<br/>
    3- difference between reference counts and insertion counts<br/>
<br/>
Align to known variants:<br/>
    GIAB project (ftp://ftp-trace.ncbi.nlm.nih.gov:/giab/ftp/release/NA12878_HG001/NISTv3.3.2/GRCh38) <br/>
    List of 15nt sequences containing variant candidates at center postion<br/>
    <br/>
Train using the high confidence calls on chromosome 21 and test on chromosome 22