package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// Struct to hold the GFCC data and associated name
type GFCCData struct {
	Name string
	Data [][]int64
}

type localdistfunc func(x []int64, y []int64) int64

func localDistanceManhattan(x []int64, y []int64) int64 {
	m := len(x)
	m2 := len(y)

	if m != m2 {
		panic("arrays have different dimensions!")
	}
	dist := int64(0)
	for i := 0; i < m; i++ {
		delta := x[i] - y[i]
		if delta < 0 {
			delta = -delta
		}
		dist += delta
	}
	return dist
}

func minInts(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func computeDTW(x [][]int64, y [][]int64, unused int64, dist localdistfunc) int64 {
	n1 := len(x)
	n2 := len(y)
	m := len(x[0])
	m2 := len(y[0])

	if m != m2 {
		panic("time series elements have different dimensions!")
	}

	dtw := make([][]int64, n2)
	for i := range dtw {
		dtw[i] = make([]int64, n1)
		for j := range dtw[i] {
			dtw[i][j] = math.MaxInt64
		}
	}
	dtw[0][0] = 0

	for j := 0; j < n2; j++ {
		for i := 0; i < n1; i++ {
			if i > 0 || j > 0 {
				cost := dist(x[i], y[j])
				min_c := int64(math.MaxInt64)
				if i > 0 {
					min_c = minInts(min_c, dtw[j][i-1])
				}
				if j > 0 {
					min_c = minInts(min_c, dtw[j-1][i])
				}
				if i > 0 && j > 0 {
					min_c = minInts(min_c, dtw[j-1][i-1])
				}
				dtw[j][i] = cost + min_c
			}
		}
	}

	return dtw[n2-1][n1-1]
}

func extractName(filename string) string {
	re := regexp.MustCompile(`^[^_]+`)
	name := re.FindString(filename)
	name = filepath.Base(name)
	return strings.ToLower(name)
}

func readCSV(filename string) ([][]int64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	rawData, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var matrix [][]int64
	for _, record := range rawData {
		var row []int64
		for _, value := range record {
			floatVal, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return nil, err
			}
			row = append(row, int64(floatVal*1e9))
		}
		matrix = append(matrix, row)
	}
	return matrix, nil
}

func writeDTWMatrix(matrix [][]int64, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	for _, row := range matrix {
		record := make([]string, len(row))
		for i, value := range row {
			record[i] = fmt.Sprintf("%d", value)
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}
	return nil
}
// --- Split into sit and stand sets ---
func runSplitDTW(datasetDir string) {
	var csvFilesA []string
	var csvFilesB []string

	// Collect files
	err := filepath.Walk(datasetDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		match, _ := regexp.MatchString("_gfcc260-26.csv$", info.Name())
		if match {
			if strings.Contains(path, "_A_") {
				csvFilesA = append(csvFilesA, path)
			} else if strings.Contains(path, "_B_") {
				csvFilesB = append(csvFilesB, path)
			}
		}
		return nil
	})
	if err != nil {
		fmt.Println("Error walking the directory:", err)
		return
	}

	fmt.Println("Found A files:", len(csvFilesA))
	fmt.Println("Found B files:", len(csvFilesB))

	// --- Load sit ---
	var gfccA []GFCCData
	for _, file := range csvFilesA {
		name := extractName(file)
		data, err := readCSV(file)
		if err != nil {
			fmt.Println("Error reading CSV file:", err)
			continue
		}
		gfccA = append(gfccA, GFCCData{Name: name, Data: data})
	}

	// --- Load stand ---
	var gfccB []GFCCData
	for _, file := range csvFilesB {
		name := extractName(file)
		data, err := readCSV(file)
		if err != nil {
			fmt.Println("Error reading CSV file:", err)
			continue
		}
		gfccB = append(gfccB, GFCCData{Name: name, Data: data})
	}

	// Allocate DTW matrices
	dtwMatrixA := make([][]int64, len(gfccA))
	for i := range dtwMatrixA {
		dtwMatrixA[i] = make([]int64, len(gfccA))
	}
	dtwMatrixB := make([][]int64, len(gfccB))
	for i := range dtwMatrixB {
		dtwMatrixB[i] = make([]int64, len(gfccB))
	}

	// Compute DTW distances for sit
	for i := 0; i < len(gfccA); i++ {
		for j := 0; j < len(gfccA); j++ {
			fmt.Printf("Comparing A %d with %d\n", i, j)
			distance := computeDTW(gfccA[i].Data, gfccA[j].Data, 0, localDistanceManhattan)
			dtwMatrixA[i][j] = distance
		}
	}

	// Compute DTW distances for stand
	for i := 0; i < len(gfccB); i++ {
		for j := 0; j < len(gfccB); j++ {
			fmt.Printf("Comparing B %d with %d\n", i, j)
			distance := computeDTW(gfccB[i].Data, gfccB[j].Data, 0, localDistanceManhattan)
			dtwMatrixB[i][j] = distance
		}
	}

	// Save results
	if err := writeDTWMatrix(dtwMatrixA, "dtw_gfcc260-26_sit.csv"); err != nil {
		fmt.Println("Error writing A matrix:", err)
	}
	if err := writeDTWMatrix(dtwMatrixB, "dtw_gfcc260-26_stand.csv"); err != nil {
		fmt.Println("Error writing B matrix:", err)
	}

	fmt.Println("✅ DTW matrices saved: dtw_gfcc260-26_sit.csv and dtw_gfcc260-26_stand.csv")
}

// --- Full set (no split) ---
func runFullDTW(datasetDir string) {
	var csvFiles []string

	// Collect files
	err := filepath.Walk(datasetDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		match, _ := regexp.MatchString("_gfcc260-26.csv$", info.Name())
		if match {
			csvFiles = append(csvFiles, path)
		}
		return nil
	})
	if err != nil {
		fmt.Println("Error walking the directory:", err)
		return
	}

	fmt.Println("Found total files:", len(csvFiles))

	// Load all embeddings
	var gfccList []GFCCData
	for _, file := range csvFiles {
		name := extractName(file)
		data, err := readCSV(file)
		if err != nil {
			fmt.Println("Error reading CSV file:", err)
			continue
		}
		gfccList = append(gfccList, GFCCData{Name: name, Data: data})
	}

	// Allocate DTW matrix
	dtwMatrix := make([][]int64, len(gfccList))
	for i := range dtwMatrix {
		dtwMatrix[i] = make([]int64, len(gfccList))
	}

	// Compute DTW distances
	for i := 0; i < len(gfccList); i++ {
		for j := 0; j < len(gfccList); j++ {
			fmt.Printf("Comparing %d with %d\n", i, j)
			distance := computeDTW(gfccList[i].Data, gfccList[j].Data, 0, localDistanceManhattan)
			dtwMatrix[i][j] = distance
		}
	}

	// Save result
	if err := writeDTWMatrix(dtwMatrix, "dtw_gfcc260-26_full.csv"); err != nil {
		fmt.Println("Error writing DTW matrix:", err)
		return
	}

	fmt.Println("✅ DTW matrix saved: dtw_gfcc260-26_full.csv")
}

// --- Main ---
func main() {
	datasetDir := "../BlowPrintData" //set to the dataset directory

	// Run split (sit and stand) version
	runSplitDTW(datasetDir)

	// Run full version
	runFullDTW(datasetDir)
}
