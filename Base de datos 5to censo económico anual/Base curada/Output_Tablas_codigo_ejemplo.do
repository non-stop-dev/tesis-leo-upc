	
	global output "G:\.shortcut-targets-by-id\1wJ8f9hhbNmaakAapfGW7H5Tmgx5Zjsb1\IPA_STORE\UPC\Tesis\"
	
	sysuse auto, clear
	
	generate gpmw = ((1/mpg)/weight)*100*1000
	
	
	gen lnM = 0
	label variable lnM "Ln(materiales)"	
	
	regress gpmw foreign
	outreg2 using "$output\resultados.xls", append se label ctitle(Total) addtext(1er comentario, NO, Robust, NO)
	
	regress gpmw foreign, vce(robust)
	outreg2 using "$output\resultados.xls", append se label ctitle(Total) addtext(1er comentario, NO, Robust, Robust, Region, Costa)
	
	regress gpmw foreign, vce(hc2)
	outreg2 using "$output\resultados.xls", append se label ctitle(Total) addtext(1er comentario, NO, Robust, hc2, Region, Sierra)

	regress gpmw foreign, vce(hc3)
	outreg2 using "$output\resultados.xls", append se label ctitle(Total) addtext(1er comentario, NO, Robust, hc3, Region, Selva)