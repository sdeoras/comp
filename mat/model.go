package mat

const (
	// model is base64 representation of protocol buffer TF model.
	// this go file was auto-generated using following command:
	//      go get github.com/sdeoras/tensorflow/cmd/mo
	//      mo u -h
	modelPB = "" +
		"CjUKB3ZlcnNpb24SBUNvbnN0KhYKBXZhbHVlEg1CCwgHEgBCBTAuMS4wKgsKBWR0eXBlEgIwBwowCgVi" +
		"dWZmMRILUGxhY2Vob2xkZXIqCwoFZHR5cGUSAjACKg0KBXNoYXBlEgQ6AhgBCjEKBnNoYXBlMRILUGxh" +
		"Y2Vob2xkZXIqCwoFZHR5cGUSAjAJKg0KBXNoYXBlEgQ6AhgBCjAKBWJ1ZmYyEgtQbGFjZWhvbGRlcioL" +
		"CgVkdHlwZRICMAIqDQoFc2hhcGUSBDoCGAEKMQoGc2hhcGUyEgtQbGFjZWhvbGRlcioLCgVkdHlwZRIC" +
		"MAkqDQoFc2hhcGUSBDoCGAEKNQoKc2hhcGVCZWdpbhILUGxhY2Vob2xkZXIqCwoFZHR5cGUSAjAJKg0K" +
		"BXNoYXBlEgQ6AhgBCjQKCXNoYXBlU2l6ZRILUGxhY2Vob2xkZXIqCwoFZHR5cGUSAjAJKg0KBXNoYXBl" +
		"EgQ6AhgBCjgKB1Jlc2hhcGUSB1Jlc2hhcGUaBWJ1ZmYxGgZzaGFwZTEqBwoBVBICMAIqDAoGVHNoYXBl" +
		"EgIwCQo/Cg1NYXRyaXhJbnZlcnNlEg1NYXRyaXhJbnZlcnNlGgdSZXNoYXBlKgcKAVQSAjACKg0KB2Fk" +
		"am9pbnQSAigACkAKCWludi9zaGFwZRIFQ29uc3QqHwoFdmFsdWUSFkIUCAMSBBICCAE6Cv//////////" +
		"/wEqCwoFZHR5cGUSAjADCj8KA2ludhIHUmVzaGFwZRoNTWF0cml4SW52ZXJzZRoJaW52L3NoYXBlKgcK" +
		"AVQSAjACKgwKBlRzaGFwZRICMAMKSgoTemVyb3MvUmVzaGFwZS9zaGFwZRIFQ29uc3QqHwoFdmFsdWUS" +
		"FkIUCAMSBBICCAE6Cv///////////wEqCwoFZHR5cGUSAjADCkwKDXplcm9zL1Jlc2hhcGUSB1Jlc2hh" +
		"cGUaBnNoYXBlMRoTemVyb3MvUmVzaGFwZS9zaGFwZSoHCgFUEgIwCSoMCgZUc2hhcGUSAjADCjwKC3pl" +
		"cm9zL0NvbnN0EgVDb25zdCoLCgVkdHlwZRICMAIqGQoFdmFsdWUSEEIOCAISADIIAAAAAAAAAAAKRAoF" +
		"emVyb3MSBEZpbGwaDXplcm9zL1Jlc2hhcGUaC3plcm9zL0NvbnN0KgcKAVQSAjACKhAKCmluZGV4X3R5" +
		"cGUSAjAJCkQKDXplcm9zXzEvc2hhcGUSBUNvbnN0Kh8KBXZhbHVlEhZCFAgDEgQSAggBOgr/////////" +
		"//8BKgsKBWR0eXBlEgIwAwo/Cgd6ZXJvc18xEgdSZXNoYXBlGgV6ZXJvcxoNemVyb3NfMS9zaGFwZSoH" +
		"CgFUEgIwAioMCgZUc2hhcGUSAjADCkkKEm9uZXMvUmVzaGFwZS9zaGFwZRIFQ29uc3QqHwoFdmFsdWUS" +
		"FkIUCAMSBBICCAE6Cv///////////wEqCwoFZHR5cGUSAjADCkoKDG9uZXMvUmVzaGFwZRIHUmVzaGFw" +
		"ZRoGc2hhcGUxGhJvbmVzL1Jlc2hhcGUvc2hhcGUqBwoBVBICMAkqDAoGVHNoYXBlEgIwAwo7CgpvbmVz" +
		"L0NvbnN0EgVDb25zdCoZCgV2YWx1ZRIQQg4IAhIAMggAAAAAAADwPyoLCgVkdHlwZRICMAIKQQoEb25l" +
		"cxIERmlsbBoMb25lcy9SZXNoYXBlGgpvbmVzL0NvbnN0KgcKAVQSAjACKhAKCmluZGV4X3R5cGUSAjAJ" +
		"CkMKDG9uZXNfMS9zaGFwZRIFQ29uc3QqHwoFdmFsdWUSFkIUCAMSBBICCAE6Cv///////////wEqCwoF" +
		"ZHR5cGUSAjADCjwKBm9uZXNfMRIHUmVzaGFwZRoEb25lcxoMb25lc18xL3NoYXBlKgcKAVQSAjACKgwK" +
		"BlRzaGFwZRICMAMKQwoScmFuZG9tX3VuaWZvcm0vbWluEgVDb25zdCoZCgV2YWx1ZRIQQg4IAhIAMggA" +
		"AAAAAAAAACoLCgVkdHlwZRICMAIKQwoScmFuZG9tX3VuaWZvcm0vbWF4EgVDb25zdCoZCgV2YWx1ZRIQ" +
		"Qg4IAhIAMggAAAAAAADwPyoLCgVkdHlwZRICMAIKZAoccmFuZG9tX3VuaWZvcm0vUmFuZG9tVW5pZm9y" +
		"bRINUmFuZG9tVW5pZm9ybRoGc2hhcGUxKgcKAVQSAjAJKgsKBWR0eXBlEgIwAioLCgVzZWVkMhICGAAq" +
		"CgoEc2VlZBICGAAKSgoScmFuZG9tX3VuaWZvcm0vc3ViEgNTdWIaEnJhbmRvbV91bmlmb3JtL21heBoS" +
		"cmFuZG9tX3VuaWZvcm0vbWluKgcKAVQSAjACClQKEnJhbmRvbV91bmlmb3JtL211bBIDTXVsGhxyYW5k" +
		"b21fdW5pZm9ybS9SYW5kb21Vbmlmb3JtGhJyYW5kb21fdW5pZm9ybS9zdWIqBwoBVBICMAIKRgoOcmFu" +
		"ZG9tX3VuaWZvcm0SA0FkZBoScmFuZG9tX3VuaWZvcm0vbXVsGhJyYW5kb21fdW5pZm9ybS9taW4qBwoB" +
		"VBICMAIKQQoKcmFuZC9zaGFwZRIFQ29uc3QqHwoFdmFsdWUSFkIUCAMSBBICCAE6Cv///////////wEq" +
		"CwoFZHR5cGUSAjADCkIKBHJhbmQSB1Jlc2hhcGUaDnJhbmRvbV91bmlmb3JtGgpyYW5kL3NoYXBlKgcK" +
		"AVQSAjACKgwKBlRzaGFwZRICMAMKQwoScmFuZG9tX25vcm1hbC9tZWFuEgVDb25zdCoZCgV2YWx1ZRIQ" +
		"Qg4IAhIAMggAAAAAAAAAACoLCgVkdHlwZRICMAIKRQoUcmFuZG9tX25vcm1hbC9zdGRkZXYSBUNvbnN0" +
		"KhkKBXZhbHVlEhBCDggCEgAyCAAAAAAAAPA/KgsKBWR0eXBlEgIwAgpxCiJyYW5kb21fbm9ybWFsL1Jh" +
		"bmRvbVN0YW5kYXJkTm9ybWFsEhRSYW5kb21TdGFuZGFyZE5vcm1hbBoGc2hhcGUxKgsKBWR0eXBlEgIw" +
		"AioLCgVzZWVkMhICGAAqCgoEc2VlZBICGAAqBwoBVBICMAkKWwoRcmFuZG9tX25vcm1hbC9tdWwSA011" +
		"bBoicmFuZG9tX25vcm1hbC9SYW5kb21TdGFuZGFyZE5vcm1hbBoUcmFuZG9tX25vcm1hbC9zdGRkZXYq" +
		"BwoBVBICMAIKRAoNcmFuZG9tX25vcm1hbBIDQWRkGhFyYW5kb21fbm9ybWFsL211bBoScmFuZG9tX25v" +
		"cm1hbC9tZWFuKgcKAVQSAjACCkIKC3JhbmRuL3NoYXBlEgVDb25zdCofCgV2YWx1ZRIWQhQIAxIEEgII" +
		"AToK////////////ASoLCgVkdHlwZRICMAMKRAoFcmFuZG4SB1Jlc2hhcGUaDnJhbmRvbV91bmlmb3Jt" +
		"GgtyYW5kbi9zaGFwZSoHCgFUEgIwAioMCgZUc2hhcGUSAjADCjoKCVJlc2hhcGVfMRIHUmVzaGFwZRoF" +
		"YnVmZjEaBnNoYXBlMSoHCgFUEgIwAioMCgZUc2hhcGUSAjAJCjoKCVJlc2hhcGVfMhIHUmVzaGFwZRoF" +
		"YnVmZjIaBnNoYXBlMioHCgFUEgIwAioMCgZUc2hhcGUSAjAJCk4KBk1hdE11bBILQmF0Y2hNYXRNdWwa" +
		"CVJlc2hhcGVfMRoJUmVzaGFwZV8yKgsKBWFkal94EgIoACoLCgVhZGpfeRICKAAqBwoBVBICMAIKMgoI" +
		"bXVsU2hhcGUSBVNoYXBlGgZNYXRNdWwqBwoBVBICMAIqDgoIb3V0X3R5cGUSAjAJCkAKCW11bC9zaGFw" +
		"ZRIFQ29uc3QqHwoFdmFsdWUSFkIUCAMSBBICCAE6Cv///////////wEqCwoFZHR5cGUSAjADCjgKA211" +
		"bBIHUmVzaGFwZRoGTWF0TXVsGgltdWwvc2hhcGUqBwoBVBICMAIqDAoGVHNoYXBlEgIwAwo6CglSZXNo" +
		"YXBlXzMSB1Jlc2hhcGUaBWJ1ZmYxGgZzaGFwZTEqBwoBVBICMAIqDAoGVHNoYXBlEgIwCQpGCgVTbGlj" +
		"ZRIFU2xpY2UaCVJlc2hhcGVfMxoKc2hhcGVCZWdpbhoJc2hhcGVTaXplKgcKAVQSAjACKgsKBUluZGV4" +
		"EgIwCQouCgVTaGFwZRIFU2hhcGUaBVNsaWNlKgcKAVQSAjACKg4KCG91dF90eXBlEgIwCQooCgxzbGlj" +
		"ZVNoYXBlT3ASCElkZW50aXR5GgVTaGFwZSoHCgFUEgIwCQpGCg9SZXNoYXBlXzQvc2hhcGUSBUNvbnN0" +
		"Kh8KBXZhbHVlEhZCFAgDEgQSAggBOgr///////////8BKgsKBWR0eXBlEgIwAwpDCglSZXNoYXBlXzQS" +
		"B1Jlc2hhcGUaBVNsaWNlGg9SZXNoYXBlXzQvc2hhcGUqBwoBVBICMAIqDAoGVHNoYXBlEgIwAwonCgdz" +
		"bGljZU9wEghJZGVudGl0eRoJUmVzaGFwZV80KgcKAVQSAjACCjoKCVJlc2hhcGVfNRIHUmVzaGFwZRoF" +
		"YnVmZjEaBnNoYXBlMSoHCgFUEgIwAioMCgZUc2hhcGUSAjAJCj4KCVJlc2hhcGVfNhIHUmVzaGFwZRoJ" +
		"UmVzaGFwZV81GgZzaGFwZTIqBwoBVBICMAIqDAoGVHNoYXBlEgIwCQpGCg9SZXNoYXBlXzcvc2hhcGUS" +
		"BUNvbnN0Kh8KBXZhbHVlEhZCFAgDEgQSAggBOgr///////////8BKgsKBWR0eXBlEgIwAwpHCglSZXNo" +
		"YXBlXzcSB1Jlc2hhcGUaCVJlc2hhcGVfNhoPUmVzaGFwZV83L3NoYXBlKgcKAVQSAjACKgwKBlRzaGFw" +
		"ZRICMAMKKQoJcmVzaGFwZU9wEghJZGVudGl0eRoJUmVzaGFwZV83KgcKAVQSAjACCjoKCVJlc2hhcGVf" +
		"OBIHUmVzaGFwZRoFYnVmZjEaBnNoYXBlMSoHCgFUEgIwAioMCgZUc2hhcGUSAjAJCjoKBFRpbGUSBFRp" +
		"bGUaCVJlc2hhcGVfOBoGc2hhcGUyKhAKClRtdWx0aXBsZXMSAjAJKgcKAVQSAjACCi8KB1NoYXBlXzES" +
		"BVNoYXBlGgRUaWxlKgcKAVQSAjACKg4KCG91dF90eXBlEgIwCQopCgt0aWxlU2hhcGVPcBIISWRlbnRp" +
		"dHkaB1NoYXBlXzEqBwoBVBICMAkKRgoPUmVzaGFwZV85L3NoYXBlEgVDb25zdCoLCgVkdHlwZRICMAMq" +
		"HwoFdmFsdWUSFkIUCAMSBBICCAE6Cv///////////wEKQgoJUmVzaGFwZV85EgdSZXNoYXBlGgRUaWxl" +
		"Gg9SZXNoYXBlXzkvc2hhcGUqBwoBVBICMAIqDAoGVHNoYXBlEgIwAwomCgZ0aWxlT3ASCElkZW50aXR5" +
		"GglSZXNoYXBlXzkqBwoBVBICMAIiAggb"
)
